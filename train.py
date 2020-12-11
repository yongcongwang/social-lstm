#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# By yongcong.wang @ 04/12/2020

import torch
import argparse

import model
import dataset


def parse_args():
    """Parse args
    python3 main.py "s" --dataset "eth" --epoch 3
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--special_file", default='', type=str)
    parser.add_argument("--special_model", default='', type=str)    
    parser.add_argument("--special_start", default=None, type=int)    
    parser.add_argument("--dataset", default="eth", type=str)
    parser.add_argument("--T_obs", default=8, type=int)
    parser.add_argument("--T_pred", default=20, type=int)    
    parser.add_argument("--epoch", default=25, type=int)
    parser.add_argument("--model_name", default="a_just_trained_model_for_")
    parser.add_argument("model_type", type=str)
    parser.add_argument("--pure_val_name", default='', type=str)

    return parser.parse_args()


def train(T_obs, T_pred, file_name, epoch_size=5):
    network = model.SocialLSTM(hidden_dim=128, mediate_dim=32, output_dim=2)

    criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(vl.parameters(), weight_decay=0.0005)

    for epoch in range(epoch_size):
        data = dataset.FrameDataset(file_name):

            for idx, data in enumerate(data):
                h = torch.zeros(data["seq"].shape[1], h_dim, device=device)
                c = torch.zeros(data["seq"].shape[1], h_dim, device=device)

                with torch.autograd.set_detect_anomaly(True):
                    Y = data["seq"][:T_pred, :, 2:].clone()
                    input_seq = data["seq"][:T_pred, :, 2:].clone()
                    coords = data["coord"][:T_pred, :, 2:].clone()
                    part_masks = data["mask"]

                    # forward propagation
                    output = network.forward(
                        input_seq, coords, part_masks, h, c, Y, T_obs, T_pred)

                    # loss
                    Y_pred = output[T_obs + 1 : T_pred]
                    Y_g = Y[T_obs + 1 : T_pred]

                    cost = criterion(Y_pred, Y_g)
                    print(epoch, idx, cost.item())

                    # backward propagation
                    optimizer.zero_grad()
                    cost.backward()
                    optimizer.step()


def main():
    args = parse_args()

    train(8, 20, "./data/eth/hotel/pixel_pos.csv")


if __name__ == "__main__":
    main()
