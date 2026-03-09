import torch
import torch.nn as nn


class ImplicitEnergyRNN(nn.Module):
    def __init__(self, energy_model, minimizer):
        """
        接收实例化的能量模型和求解器以初始化
        :param energy_model:能量模型输入
        :param minimizer:求解器输入
        """

    def forward(self, x_sequence, h_0):
        """

        :param x_sequence: 整个输入序列 $x_1, \dots, x_T$)
        :param h_0:初始隐变量
        :return:h_sequence (所有时间步的隐状态集合), y_sequence (网络读出序列)
        """