
class EnergyMinimizer(object):
    """
    能量最小值寻找器，利用梯度下降
    """
    def __init__(self, energy_model):
        """
        初始化求解器，输入一个需要优化的能量模型，也输入超参数
        :param energy_model: 能量模型输入
        """

    def solve(self, h_prev, x_t, h_init_guess):
        """
        最小能量求解，寻找使能量最小的 $h_t$
        :param h_prev:上一个时刻的隐状态输入
        :param x_t:当前时刻的外输入
        :param h_init_guess:
        :return: 当前状态的初值猜测
        """

