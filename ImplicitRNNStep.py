import torch


class ImplicitRNNStep(torch.autograd.Function):
    def forward(ctx, energy_model, h_optimal, h_prev, x_t, *theta_params):
        """
        PyTorch 的前向传递, 只负责把参与反向计算的张量保存下来。
        :param energy_model:
        :param h_optimal:
        :param h_prev:
        :param x_t:
        :param theta_params:
        :return:
        """

    def backward(ctx, grad_h_optimal):
        """
        接收来自未来时间步或Loss的梯度
        grad_h_optimal($a_t$)，利用公式
        $\frac{\partial h_t}{\partial \theta} = -(\frac{\partial ^ 2 E_t}{\partial h_t ^ 2}) ^ {-1}[\dots]$
        计算并返回$h_prev$对$\theta$ 的梯度 。
        :param grad_h_optimal:
        :return:
        """
        