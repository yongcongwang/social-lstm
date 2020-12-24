#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# By yongcong.wang @ 04/12/2020

import torch
import argparse
from torch.utils.tensorboard import SummaryWriter

import model
import dataset
from visualization import Visualization


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
    parser.add_argument("--pure_val_name", default='', type=str)

    return parser.parse_args()


def train(T_obs, T_pred, file_names, epoch_size=50):
    print("-------------------------------------------")
    writer = SummaryWriter("./log/loss")
    vis = Visualization()
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    network = model.SocialLstm(hidden_dim=128, mediate_dim=32, output_dim=2)
    network.to(device)
    #network = torch.load("./log/model/559_model.pt")
    #network.eval()
    #network.to(device)

    criterion = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(network.parameters(), weight_decay=0.0005)

    loss = 0.
    for epoch in range(epoch_size):
        cost_sum = 0.
        cost_cnt = 0
        for file_name in file_names:
            file_data = dataset.FrameDataset(file_name)
            print(file_name)
            for (idx, data) in enumerate(file_data):
                h = torch.zeros(data["ped_trajs"].shape[1], 128, device=device)
                c = torch.zeros(data["ped_trajs"].shape[1], 128, device=device)
                optimizer.zero_grad()

                with torch.autograd.set_detect_anomaly(True):
                    Y = data["ped_trajs"][:T_pred, :, 1:].clone()
                    trajs = data["ped_trajs"][:T_pred, :, 1:].clone()
                    traj_masks = data["ped_masks"]

                    # forward propagation
                    output = network.forward(
                        trajs, traj_masks, h, c, Y, T_obs, T_pred)

                    # loss
                    Y_pred = output[T_obs + 1 : T_pred]
                    Y_g = Y[T_obs + 1 : T_pred]

                    cost = criterion(Y_pred, Y_g)
                    cost_sum += cost.item()
                    cost_cnt += 1
                    if cost_cnt % 50 == 0:
                        vis.plot(
                            Y[:T_pred, :, :].clone().detach().cpu().tolist(),
                            output[:T_pred, :, :].clone().detach(
                                ).cpu().tolist(),
                            T_obs, T_pred
                            )

                    # backward propagation
                    cost.backward()
                    optimizer.step()

        loss = cost_sum / cost_cnt
        print("epoch: ", epoch, "loss: ", loss)
        writer.add_scalar("Loss/train", loss)

    torch.save(network, "./log/model/" + str(int(loss * 100)) + "_model.pt")
    writer.close()


def main():
    args = parse_args()

    file_names = [
        "./data/eth/hotel/pixel_pos_interpolate.csv",
        "./data/eth/univ/pixel_pos_interpolate.csv",
        "./data/ucy/univ/pixel_pos_interpolate.csv",
        "./data/ucy/zara/zara01/pixel_pos_interpolate.csv",
        #"./data/ucy/zara/zara02/pixel_pos_interpolate.csv"
        ]
    train(8, 20, file_names)


if __name__ == "__main__":
    main()
