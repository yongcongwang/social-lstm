#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# By yongcong.wang @ 27/11/2020

import torch
import torch.nn as nn

class SocialLstm(nn.Module):
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

        super(SocialLstm, self).__init__()

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
        self.SocialEmbedding = nn.Linear((self.N_size + 1) ** 2 * hidden_dim,
                                         social_dim)
        self.LSTMCell = nn.LSTMCell(mediate_dim + social_dim, hidden_dim)
        self.OutputLayer = nn.Linear(hidden_dim, output_dim)
        self.Phi = Phi(dropout_prob=dropout_prob)
        self.CorrNormLayer = nn.Sigmoid()

    def social_pooling(self, h_tm1, coords, mask):
        with torch.no_grad():

            margin_thick = 2 * self.N_size * self.grid_cell_size
            x_min = torch.min(coords[:, 0]) - margin_thick
            x_max = torch.max(coords[:, 0]) + margin_thick
            y_min = torch.min(coords[:, 1]) - margin_thick
            y_max = torch.max(coords[:, 1]) + margin_thick
            lt_corner = torch.tensor([x_min, y_min], device=self.device)

            positions = [[int((coords[i][0] - x_min) // self.grid_cell_size),
                          int((coords[i][1] - y_min) // self.grid_cell_size)]
                         for i in range(len(coords))]
            positions = [[pos[0] * mask[i], pos[1] * mask[i]]
                         for i, pos in enumerate(positions)]

            grid_width = int((x_max - x_min) // self.grid_cell_size)
            grid_height = int((y_max - y_min) // self.grid_cell_size)
            grid_htm1 = torch.zeros(grid_width, grid_height, self.hidden_dim,
                                    device=self.device)
            for idx in range(coords.shape[0]):
                x = int(positions[idx][0])
                y = int(positions[idx][1])
                grid_htm1[x][y] = grid_htm1[x][y] + h_tm1[idx]

            # calc H
            H = torch.zeros(coords.shape[0], self.N_size + 1, self.N_size + 1,
                            self.hidden_dim, device=self.device)
            for idx in range(coords.shape[0]):
                if mask[idx] == 0:
                    continue
                x = positions[idx][0]
                y = positions[idx][1]
                R = self.grid_cell_size * self.N_size / 2
                H[idx] = grid_htm1[int(x - R):int(x + R),
                                   int(y - R):int(y + R), :]
            H = H.reshape(coords.shape[0],
                          (self.N_size + 1) ** 2 * self.hidden_dim)

        return H

    def forward(self, X, part_masks, all_h_t, all_c_t, Y, T_obs,
                T_pred):
        outputs = torch.empty(X.shape[0], X.shape[1], self.output_dim,
                              device=self.device)

        last_point = X[T_obs + 1, :].clone()
        for frame_idx, x in enumerate(X):
            if frame_idx <= T_obs:
                # input embedding
                r = self.Phi(self.InputEmbedding(x))
                # social pooling embedding
                H = self.social_pooling(all_h_t, X[frame_idx],
                                        part_masks[frame_idx])
                # hidden state embedding
                e = self.Phi(self.SocialEmbedding(H))
                concat_embed = torch.cat((r, e), 1)
                all_h_t, all_c_t = self.LSTMCell(concat_embed,
                                                 (all_h_t, all_c_t))
                part_mask = torch.t(part_masks[frame_idx]) \
                                 .view(part_masks[frame_idx].shape[0], 1) \
                                 .expand(part_masks[frame_idx].shape[0],
                                         self.output_dim)
                outputs[frame_idx] = self.OutputLayer(all_h_t) * part_mask
            elif T_obs < frame_idx <= T_pred:
                # input embedding
                #r = self.Phi(self.InputEmbedding(last_offs))
                r = self.Phi(self.InputEmbedding(last_point))
                # social pooling embedding
                H = self.social_pooling(all_h_t, X[frame_idx],
                                        part_masks[frame_idx])
                e = self.Phi(self.SocialEmbedding(H))
                concat_embed = torch.cat((r, e), 1)
                all_h_t, all_c_t = self.LSTMCell(concat_embed,
                                                 (all_h_t, all_c_t))
                part_mask = torch.t(part_masks[frame_idx]) \
                                 .view(part_masks[frame_idx].shape[0], 1) \
                                 .expand(part_masks[frame_idx].shape[0],
                                         self.output_dim)
                outputs[frame_idx] = self.OutputLayer(all_h_t) * part_mask
                last_point = outputs[frame_idx - 1].clone()
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
