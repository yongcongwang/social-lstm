#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# By yongcong.wang @ 21/12/2020

import matplotlib.pyplot as plt
import time

class Visualization:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure()

    def plot(self, xy_real, xy_pred, t_obs, t_pred):
        self.fig.clear()
        real_xs = []
        real_ys = []
        for idx in range(len(xy_real[0])):
            x_traj = [frame[idx][0] for frame in xy_real]
            y_traj = [frame[idx][1] for frame in xy_real]
            x_unique = []
            y_unique = []
            for i in range(len(x_traj)):
                if abs(x_traj[i]) > 0.001 and abs(y_traj[i]) > 0.001:
                    x_unique.append(x_traj[i])
                    y_unique.append(y_traj[i])
            real_xs.append(x_unique)
            real_ys.append(y_unique)

        pred_xs = []
        pred_ys = []
        for idx in range(len(xy_pred[0])):
            x_traj = [frame[idx][0] for frame in xy_pred]
            y_traj = [frame[idx][1] for frame in xy_pred]
            x_unique = []
            y_unique = []
            for i in range(len(x_traj)):
                if abs(x_traj[i]) > 0.00001 and abs(y_traj[i]) > 0.00001:
                    x_unique.append(x_traj[i])
                    y_unique.append(y_traj[i])
            pred_xs.append(x_unique)
            pred_ys.append(y_unique)

        for idx, xs in enumerate(real_xs):
            plt.plot(xs[:t_obs], real_ys[idx][:t_obs], "r--")
            plt.plot(xs[t_obs:], real_ys[idx][t_obs:], "r")

        for idx, xs in enumerate(pred_xs):
            plt.plot(xs[t_obs:], pred_ys[idx][t_obs:], "b")

        print("traj_len: ", [len(xs) for xs in real_xs])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
