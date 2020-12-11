#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# By yongcong.wang @ 27/11/2020

import torch
import torch.nn as nn

import numpy as np
from torch.autograd import Variable


class SocialLstm(nn.Model):
    """Social LSTM class

    Implementation of Social LSTM
    https://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf
    """

    def __init__(self, input_dim=2, hidden_dim=20, mediate_dim=128,
                 output_dim=2, social_dim=16, traj_num=3, dropout_prob=0.0,
                 N_size=2, grid_cell_size=0.3):
        """Initializer function
        Use init params to build social-lstm
        """

        super(SocaiLSTM, self).__init_()

        # basic params
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim 
        self.mediate_dim = mediate_dim
        self.output_dim = output_dim
        self.traj_num = traj_num
        self.grid_cell_size = grid_cell_size
        self.N_size = N_size if N_size % 2 == 0 else N_size + 1
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        # embedding layers
        self.InputEmbedding = nn.Linear(input_dim, mediate_dim)
        self.SocialEmbedding = nn.Linaer((self.N_size+1)**2 * hidden_dim,
                                         social_dim)
        self.LSTMCell = nn.LSTMCell(mediate_dim+social_dim, hidden_dim)
        self.OutputLayer = nn.Linear(hidden_dim, output_dim)
        self.Phi = Phi(dropout_prob=dropout_prob)
        self.CorrNormLayer = nn.Sigmoid()

    def social_pooling(self, h_tm1, coords, mask):
        with torch.no_grad():
            H = torch.zeros(coords.shape[0], self.N_size+1, self.N_size+1,
                            self.hidden_dim, device=self.device)

            margin_thick = 2 * self.N_size * self.grid_cell_size
            x_min = torch.min(coords[:,0]) - margin_thick
            x_max = torch.max(coords[:,0]) + margin_thick
            y_min = torch.min(coords[:,1]) - margin_thick
            y_max = torch.max(coords[:,1]) + margin_thick
            lt_corner = torch.tensor([x_min, y_max], device=self.device)

            POS = [[int(xy) for xy in
                    (coords[idx] - lt_corner) // self.grid_cell_size] if
                        mask[idx] != 0 else [0,0]
                            for idx in range(coords.shape[0])]
            h_tm1_masked = mask.clone().view(mask.shape[0], 1).expand(
                mask.shape[0], self.hidden_dim) * h_tm1.clone()

            grid_width = int((x_max - x_min) // self.grid_cell_size)
            grid_height = int((y_max - y_min) // self.grid_cell_size)
            grid_htm1 = torch.zeros(grid_width, grid_height, device=self.device)
            for idx in range(coords.shape[0]):
                grid_htm1[POS[idx][0]][POS[[idx][1]]] += h_tm1[idx]

            # calc H
            for idx in range(coords.shape[0]):
                if mask[idx] == 0:
                    continue
                x = POS[idx][0]
                y = POS[idx][1]
                R = self.grid_cell_size * self.N_size / 2
                H[idx] = grid_htm1[int(x - R):int(x + R),
                                   int(y - R):int(y + R), :]
            H = H.reshape(coords.shape[0],
                          (self.N_size + 1) ** 2 * self.hidden_dim)

        return H

    def forward(self, X, coords, part_masks, all_h_t, all_c_t, Y, T_obs,
                T_pred):
        outputs = torch.empty(X.shape[0], X.shape[1], self.output_dim,
                              device=self.device)

        last_points = coords[T_obs + 1, :]

        for frame_idx, (x, coord) in enumerate(zip(X, coords)):
            if frame_idx <= T_obs:
                # input embedding
                r = self.Phi(self.InputEmbedding(x))
                # social pooling embedding
                H = self.social_pooling(all_h_t, coords,
                                        part_masks[frame_idx][0])
                # hidden state embedding
                e = self.Phi(self.SocialEmbedding(H))
                concat_embed = torch.cat((r, e), 1)
                all_h_t, all_c_t = self.LSTMCell(concat_embed,
                                                 (all_h_t, all_c_t))
                part_mask = torch.t(part_masks[frame_idx]).expand(
                    part_masks[frame_idx].shape[1], self.output_dim)
                outputs[frame_idx] = self.OutputLayer(all_h_t) * part_mask
            elif T_obs < frame_idx <= T_pred:
                last_offs = outputs[frame_idx - 1].clone()
                for idx in range(last_points.shape[0]):
                    last_points[idx] += last_offs[idx]
                last_points += last_offs

                # input embedding
                r = self.Phi(self.InputEmbedding(last_offs))
                # social pooling embedding
                H = self.socialPooling(all_h_t, coords,
                                       part_masks[frame_idx][0])
                e = self.Phi(self.SocialEmbedding(H))
                concat_embed = torch.cat((r, e), 1)
                all_h_t, all_c_t = self.LSTMCell(concat_embed,
                                                 (all_h_t, all_c_t))
                part_mask = torch.t(part_masks[frame_idx]).expand(
                    part_masks[frame_idx].shape[1], self.output_dim)
                outputs[frame_idx] = self.OutputLayer(all_h_t) * part_mask
            else:
                outputs[frame_idx] = torch.zeros(X.shape[1], self.output_dim)

        return outputs


class Phi(nn.Module):
    """a non-linear layer
    """
    def __init__(self, dropout_prob):
        super(Phi, self).__init__()
        self.dropout_prob = dropout_prob
        self.ReLU = nn.ReLU()
        self.Dropout = nn.Dropout(p=dropout_prob)
    
    def forward(self, x):
        return self.Dropout(self.ReLU(x))
