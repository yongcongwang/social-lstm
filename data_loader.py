#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# By yongcong.wang @ 24/12/2020

import os
import pickle
import numpy as np

class DataLoader():

    def __init__(self, batch_size=50, seq_length = 5, datasets=[],
                 dataset_selections=[0, 1, 2, 3, 4], is_process_raw_data=False,
                 infer=False):
        """
        Initializer for class
        params:
        batch_size : Size of mini-batch
        seq_length : Sequence length to be considered
        datasets : All dataset names
        dataset_selections : ith dataset to be considered
        is_process_raw_data : If to process the raw data again
        """
        self.datasets = datasets
        self.used_datasets = [datasets[i] for i in dataset_selections]
        self.infer = infer

        self.pkl_dir = "./data"

        self.batch_size = batch_size
        self.seq_length = seq_length

        self.val_perception = 0.2

        pkl_file = os.path.join(self.pkl_dir, "trajectories.pkl")

        if not(os.path.exists(pkl_file)) or is_process_raw_data:
            print(f"Creating pre-processed data {pkl_file} from raw data")
            self.frame_preprocess(self.used_datasets, pkl_file)

        self.load_preprocessed(pkl_file)
        self.reset_batch_pointer(valid=False)
        self.reset_batch_pointer(valid=True)

    def frame_preprocess(self, data_dirs, data_file):
        """
        Transpose csv data to pkl file
        """

    def load_preprocessed(self, data_file):
        pass

    def reset_batch_pointer(self, valid=False):
        """
        Reset all pointers
        """
        if not valid:
            # Go to the first frame of the first dataset
            self.dataset_pointer = 0
            self.frame_pointer = 0
        else:
            self.valid_dataset_pointer = 0
            self.valid_frame_pointer = 0


if __name__ == "__main__":
    datasets = [
            "./data/eth/univ/pixel_pos_interpolate.csv",
            "./data/eth/hotel/pixel_pos_interpolate.csv",
            "./data/ucy/univ/pixel_pos_interpolate.csv",
            "./data/ucy/zara/zara01/pixel_pos_interpolate.csv",
            "./data/ucy/zara/zara02/pixel_pos_interpolate.csv"
            ]
    dataloader = DataLoader(datasets=datasets, is_process_raw_data=True)
