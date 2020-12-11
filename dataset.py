#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# By yongcong.wang @ 08/12/2020

import torch.utils.data.Dataset


class FrameDataset(Dataset):
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

    def read_csv(self, path):
        """Read csv data from file

        Read csv data from file and ~Translate data
        from 
        [[frame_num0,1,2...], [pedestrian_id0,1,2...], [y0,1,2...], [x0,1,2...]]
        to
        [[frame_num0, pedestrian_id0, x0, y0],
         [frame_num1, pedestrian_id1, x1, y1],
         [frame_num2, pedestrian_id2, x2, y2],
         ...
         ]
        """
        file_data = []
        with open(path, "r") as file:
            line_data = [int(float(data)) if i < 2 else float(data) for
                         i, data in enumerate(line.rsplit())]
            line_data[2], line_data[3] = line_data[3], line_data[2]
            file_data.append(line_data)

        return file_data

    def extract_data(self, data, length):
        """Generate data_packet for each pedestrian with time longer than length
        return all packets
        """
        file_data = deepcopy(data)
        # sort by frame number
        file_data = sorted(file_data, key=lambda data : data[0])
        time_step = 0
        for i in range(len(file_data) - 1):
            time_step = file_data[i+1][0] - file_data[i][0]
            if timestamp != 0:
                break

        # search pedestrian with the trajectory time longer than length
        time_stamps = self.get_length_timestamps(file_data, length)

        # generate data_packet for each pedestrian got in above step.
        data_packets = []
        for (traj_idx, t_s, t_e) in time_stamps:
            traj_data = self.get_data_in_time(
                data, t_s, t_s + (length + 1) * time_step)
            traj_list, participant_masks, off_data, coord_data =
                self.generate_batch_data(traj_data);
            data_packet = {
                "traj_list" : traj_list,
                "mask" : participant_masks,
                "seq" : off_data,
                "coord" : coord_data
                }
            data_packets.append(data_packet)

        return data_packets

    def get_length_timestamps(self, data, length):
        """Get pedestrian's id with the duration bigger than length
        return: [pedestrian_id, start_time, end_time]
        """
        file_data = deepcopy(data)
        file_data = sorted(file_data, key=lambda data : data[1])

        time_stamps = []
        temp_traj_idx = file_data[0][1]
        cnt = 0

        (t_s, t_e) = (file_data[0][0], file_data[0][0])

        for i, line in enumerate(file_data):
            if line[1] == temp_traj_idx:
                cnt += 1
                continue
            if cnt >= length:
                t_e = line[0]
                time_stamps.append((temp_traj_idx, t_s, file_data[i-1][0]))
            temp_traj_idx = line[1]
            cnt = 0
            t_s = line[0]

        return time_stamps

    def get_data_in_time(self, data, t_s, t_e):
        """Extract data between start_time and end_time
        """
        file_data = deepcopy(data)
        file_data = sorted(file_data, key=lambda data : data[0])

        return [line if t_s <= line[0] <= t_e for line in file_data]

    def generate_batch_data(self, data):
        file_data = sorted(data, key=lambda data : data[1])  # by ped_id
        file_data_sort = sorted(data, key=lambda data : data[0])  # by frame

        ped_id_list, frame_masks, coord_tensors =
            self.text_to_frame_tensor(file_data_sort)
        
        # use position(x,y) offset, not the global coord
        file_data_off = []
        for i, line in enumerate(file_data):
            if i > 0 and if file_data[i][1] == file_data[i-1][1]:
                file_data_off.append([file_data[i - 1][0], file_data[i - 1][1],
                                      file_data[i][2] - file_data[i - 1][2],
                                      file_data[i][3] - file_data[i - 1][3]])
        file_data_off.sort(key=lambda data : data[0])        
        
        ped_id_list, participant_masks, offset_tensors =
            self.tex_to_frame_tensor(file_data_off)

        return ped_id_list, frame_masks, offset_tensors, coord_tensors

    def text_to_frame_tensor(self, data):
        """Turn data to a list of pedestrian id and trajectory in each frame
        return 
        ped_id_list: a list of all ids of pedestrians in this dataset
        frame_mask: a list of available pedestrian's id in each frame
        frame_tensors: a list of pedestrian id and trajectory in each frame
        """
        # unique list
        ped_id_list = list(set([line[1] for line in data])).sort()
        frame_list = list(set([line[0] for line in data])).sort()

        # split line to groups with same frame
        data_temp = []
        frames = []
        last_frame = data[0][0]
        for line in data:
            if line[0] == frame_num:
                data_temp.append(line)
                continue
            data_temp.sort(key=lambda data : data[1])
            frames.append(data_temp)
            data_temp = [line]
            frame_num = line[0]

        # get pedestrian(traj) ids in each frame
        frame_trajs = [[]] * len(frames)
        for frame_idx, line in enumerate(frames):
            for traj_idx, traj in enumerate(ped_id_list):
                in_flag = False
                for data in line:
                    if data[1] == traj:
                        in_flag = True
                        frame_trajs[frame_idx].append(
                            ped_id_list.index(data[1]))
                if not in_flag:
                    frames[frame_idx].append(
                        [frame_list[frame_idx], traj, 0., 0.])
            frames[frame_idx].sort(key=lambda data : data[1])

        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        frame_tensors = torch.tensor(frames, device=device)

        # mask the pedestrian(traj) ids in ped_id_list for each frame
        frame_masks = []
        for frame_idx, line in enumerate(frame_trajs):
            fram_masks.append(
                [[torch.tensor(1.) if i in frame_trajs[frame_idx] else
                  torch.tensor(0.) for i in range(len(ped_id_list))]])
        frame_masks = torch.tensor(frame_masks, device=device)
        
        return ped_id_list, frame_masks, frame_tensors
