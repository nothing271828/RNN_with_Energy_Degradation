import torch
import torch.nn as nn


class EnergyModel(nn.Module):
    """
    纯粹的数学定义层，实现能量函数 $E(h_t, h_{t - 1}, x_t; \theta)$ 以及输出函数 $y_t = \phi_{out}(h_t)$ 。
    """

    def __init__(self):
        """
        初始化网络参数
        """
        super(EnergyModel, self).__init__()

    def compute_energy(self, h_current, h_prev, x_t):
        """
        能量函数计算
        :param h_current: 当前的隐状态
        :param h_prev: 上一个时刻的隐状态
        :param x_t: 当前时刻外输入
        :return: 系统能量
        """

    def readout(self, h_current):
        """
        读出函数，输入隐状态
        :param h_current: 隐状态输入
        :return: 输出
        """
