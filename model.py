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
    def __init__(self, args):
        """Initializer function
        Use input args to build network
        """
        self.use_cuda = args.use_cuda

        self.input_size = args.input_size
        self.output_size = args.output_size

        self.embedding_size = args.embedding_size
        self.rnn_size = args.rnn_size

        # LSTM
        self.cell = nn.LSTMCell(2 * self.embedding_size, self.rnn_size)
        
    def forward(self, input_data, grids, hidden_states,
                cell_states, ped_list):


