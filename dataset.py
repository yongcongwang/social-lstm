#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# By yongcong.wang @ 08/12/2020

import torch.utils.data
from copy import deepcopy
import numpy as np


class FrameDataset(torch.utils.data.Dataset):
    """Dataset to process csv file
    """
    def __init__(self, path, length=20):
        """Initialization
        """

        csv_data = self.read_csv(path)
        self.data_packets = self.extract_data(csv_data, length)

    def __len__(self):
        """Length of data items
        """
        return len(self.data_packets)

    def __getitem__(self, idx):
        """Get item by index
        """
        return self.data_packets[idx]

    def extract_data(self, csv, length):
        """Generate data_packet for each pedestrian with time longer than length
        return all packets
        """

        #step = self.get_frame_step(csv)
        step = 10
        # search pedestrian with the trajectory time longer than length
        pedestrians = self.get_pedestrians_of_length(csv, length * step)

        # generate data_packet for each pedestrian got in above step.
        data_packets = []
        for (traj_idx, t_s, t_e) in pedestrians:
            traj_data = self.get_data_in_time(csv, t_s, t_e, step)
            traj_data = sorted(traj_data, key=lambda data : data[0])  # by frame
            ped_ids, ped_masks, ped_trajs, = \
                self.text_to_frame_tensor(traj_data)

            data_packet = {
                "ped_ids" : deepcopy(ped_ids),
                "ped_masks" : deepcopy(ped_masks),
                "ped_trajs" : deepcopy(ped_trajs),
                }
            data_packets.append(data_packet)

        return data_packets

    def read_csv(self, path):
        """Read csv data from file

        Read csv data from file and translate data from string to value
        [[frame_num0,1,2...], [pedestrian_id0,1,2...], [y0,1,2...], [x0,1,2...]]
        """
        with open(path, "r") as csv:
            file_data = [[float(data) for data in line.rsplit(",")]
                         for line in csv]
            file_data[0] = [int(ele) for ele in file_data[0]]
            file_data[1] = [int(ele) for ele in file_data[1]]
            file_data[2], file_data[3] = file_data[3], file_data[2]
            return file_data

        return []

    def get_frame_step(self, data):
        """Frame step
        """
        frames = deepcopy(data[0])
        frames = sorted(list(set(frames)))
        frame_step = min([ele - frames[i - 1] if i > 0 else 100
                          for i, ele in enumerate(frames)])

        return frame_step

    def get_pedestrians_of_length(self, csv, length):
        """Get pedestrian's id with the duration bigger than length
        return: [pedestrian_id, start_time, end_time]
        """
        data = deepcopy(csv[0:2])
        data = sorted(np.array(data).T.tolist(), key=lambda line : line[1])

        left = 0
        right = 0
        res = []
        while right < len(data):
            if data[left][1] == data[right][1]:
                right += 1
                continue
            frames = [data[i][0] for i in range(left, right)]
            if (max(frames) - min(frames)) > length:
                res.append([data[left][1], min(frames), max(frames)])

            left = right
            right += 1

        return res

    def get_data_in_time(self, csv, t_s, t_e, step):
        """Extract data between start_time and end_time
        """
        data = deepcopy(csv)
        data = np.array(data).T.tolist()
        data = sorted(data, key=lambda line : line[0])

        res = []
        for line in data:
            if t_s <= line[0] <= t_e and int((line[0] - t_s) % step) == 0:
                res.append(line)
        return res

    def text_to_frame_tensor(self, data):
        """Turn data to a list of pedestrian id and trajectory in each frame
        return 
        ped_id_list: a list of all ids of pedestrians in this dataset
        frame_mask: a list of available pedestrian's id in each frame
        frame_tensors: a list of pedestrian id and trajectory in each frame
        """
        # split line to groups with same frame
        left = 0
        right = 0
        frames = []
        while right < len(data):
            if int(data[left][0]) == int(data[right][0]):
                right += 1
                continue
            curr_frames = sorted([data[i] for i in range(left, right)],
                                 key=lambda line : line[1])
            frames.append(curr_frames)

            left = right
            right += 1

        # unique list
        ped_list = sorted(list(set([int(line[1]) for line in data])))
        patch_frames = [[[ped, 0., 0.] for ped in ped_list] for f in frames]
        mask_frames = [[0. for ped in ped_list] for f in frames]
        for frame_idx, frame in enumerate(patch_frames):
            frame_ped_list = [int(ped[1]) for ped in frames[frame_idx]]
            for ped_idx, ped in enumerate(ped_list):
                if ped in frame_ped_list:
                    index = frame_ped_list.index(ped)
                    frame[ped_idx][1] = frames[frame_idx][index][2]
                    frame[ped_idx][2] = frames[frame_idx][index][3]
                    mask_frames[frame_idx][ped_idx] = 1.

        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        frame_tensors = torch.tensor(patch_frames, device=device)
        frame_masks = torch.tensor(mask_frames, device=device)
        
        return ped_list, frame_masks, frame_tensors

if __name__ == "__main__":
    file_names = [
        "./data/eth/hotel/pixel_pos_interpolate.csv",
        "./data/eth/univ/pixel_pos_interpolate.csv",
        "./data/ucy/univ/pixel_pos_interpolate.csv",
        "./data/ucy/zara/zara01/pixel_pos_interpolate.csv",
        #"./data/ucy/zara/zara02/pixel_pos_interpolate.csv"
        ]

    for file_name in file_names:
        file_data = FrameDataset(file_name)
        for line in file_data:
            print(line["ped_masks"])
            break;
