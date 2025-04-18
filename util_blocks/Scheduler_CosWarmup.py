from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


def get_position_from_periods(iteration, cumulative_period):
    """Get the position from a period list.

    It will return the index of the right-closest number in the period list.
    For example, the cumulative_period = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 2.

    Args:
        iteration (int): Current iteration.
        cumulative_period (list[int]): Cumulative period list.

    Returns:
        int: The position of the right-closest number in the period list.
    """
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i
    # 如果迭代次数超过了所有周期的总和，返回最后一个周期的索引
    return len(cumulative_period) - 1


class CosineAnnealingRestartCyclicLR(_LRScheduler):
    """ Cosine annealing with restarts learning rate scheme.
    An example of config:
    periods = [10, 10, 10, 10]
    restart_weights = [1, 0.5, 0.5, 0.5]
    eta_min=1e-7
    It has four cycles, each has 10 iterations. At 10th, 20th, 30th, the
    scheduler will restart with the weights in restart_weights.
    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        periods (list): Period for each cosine anneling cycle.
        restart_weights (list): Restart weights at each restart iteration.
            Default: [1].
        eta_min (float): The mimimum lr. Default: 0.
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(self,
                 optimizer,
                 periods,
                 restart_weights=(1,),
                 eta_mins=(0,),  # 0 -> 1e-9
                 last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_mins = eta_mins
        assert (len(self.periods) == len(self.restart_weights)
                ), 'periods and restart_weights should have the same length.'
        self.cumulative_period = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]
        super(CosineAnnealingRestartCyclicLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch,
                                        self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]
        eta_min = self.eta_mins[idx]

        return [
            eta_min + current_weight * 0.5 * (base_lr - eta_min) *
            (1 + math.cos(math.pi * (
                    (self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]


class CosineAnnealingRestartLR(_LRScheduler):
    """ Cosine annealing with restarts learning rate scheme.

    An example of config:
    periods = [10, 10, 10, 10]
    restart_weights = [1, 0.5, 0.5, 0.5]
    eta_min=1e-7

    It has four cycles, each has 10 iterations. At 10th, 20th, 30th, the
    scheduler will restart with the weights in restart_weights.

    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        periods (list): Period for each cosine anneling cycle.
        restart_weights (list): Restart weights at each restart iteration.
            Default: [1].
        eta_min (float): The mimimum lr. Default: 0.
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(self, optimizer, periods, restart_weights=(1,), eta_min=0, last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_min = eta_min
        assert (len(self.periods) == len(
            self.restart_weights)), 'periods and restart_weights should have the same length.'
        self.cumulative_period = [sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))]
        super(CosineAnnealingRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch, self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]

        return [
            self.eta_min + current_weight * 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * ((self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]


# 在文件末尾添加以下代码
if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import numpy as np

    # 创建一个简单的模型
    model = nn.Sequential(nn.Linear(10, 1))
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 设置余弦退火学习率调度器
    periods = [30, 30, 30]
    restart_weights = [1, 0.5, 0.5]
    eta_min = 1e-7
    cosine_scheduler = CosineAnnealingRestartLR(
        optimizer, periods=periods, restart_weights=restart_weights, eta_min=eta_min
    )

    # 设置预热学习率调度器
    warmup_epochs = 10
    warmup_scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1.0, total_epoch=warmup_epochs, after_scheduler=cosine_scheduler
    )

    # 记录学习率变化
    lr_warmup_cosine = []
    lr_cosine = []

    # 模拟训练过程
    max_epochs = sum(periods) + 10  # 额外添加一些epoch以便观察

    # 创建另一个优化器用于比较纯余弦退火
    optimizer_cosine = optim.Adam(model.parameters(), lr=0.01)
    cosine_only_scheduler = CosineAnnealingRestartLR(
        optimizer_cosine, periods=periods, restart_weights=restart_weights, eta_min=eta_min
    )

    # 模拟训练过程并记录学习率
    for epoch in range(max_epochs):
        # 模拟训练步骤（在实际训练中，这里会有前向传播、反向传播等）
        # 先执行优化器步骤
        optimizer.step()
        optimizer_cosine.step()

        # 然后更新学习率
        warmup_scheduler.step()
        cosine_only_scheduler.step()

        # 记录学习率
        lr_warmup_cosine.append(optimizer.param_groups[0]['lr'])
        lr_cosine.append(optimizer_cosine.param_groups[0]['lr'])

    # 绘制学习率变化曲线
    plt.figure(figsize=(10, 5))
    epochs = list(range(max_epochs))

    plt.plot(epochs, lr_warmup_cosine, 'b-', label='Warmup + Cosine Annealing')
    plt.plot(epochs, lr_cosine, 'r--', label='Cosine Annealing Only')

    # 标记预热阶段
    plt.axvline(x=warmup_epochs, color='g', linestyle='--', label='End of Warmup')

    # 标记每个周期的结束
    cumulative_periods = np.cumsum(periods)
    for i, period_end in enumerate(cumulative_periods):
        plt.axvline(x=period_end, color='k', linestyle=':', alpha=0.5)
        if i < len(cumulative_periods) - 1:
            plt.text(period_end + 0.5, max(lr_warmup_cosine) * 0.8, f'Restart {i + 1}',
                     rotation=90, verticalalignment='center')

    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Cosine Annealing with Warmup Learning Rate Schedule')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图像
    # plt.savefig('cosine_warmup_lr_schedule.png', dpi=300, bbox_inches='tight')
    plt.show()