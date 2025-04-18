import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        """
        通道注意力机制

        参数:
            in_channels (int): 输入特征图的通道数
            reduction_ratio (int): 降维比例，用于减少参数量
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享 MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化分支
        avg_out = self.fc(self.avg_pool(x))

        # 最大池化分支
        max_out = self.fc(self.max_pool(x))

        # 融合两个分支
        out = avg_out + max_out

        # 应用 sigmoid 激活函数得到注意力权重
        attention = self.sigmoid(out)

        # 将注意力权重应用到输入特征
        return x * attention


if __name__ == "__main__":
    # 测试代码
    # 创建一个随机的输入张量，模拟一个批次的特征图
    batch_size = 4
    channels = 64
    height = 32
    width = 32

    # 创建输入张量
    x = torch.randn(batch_size, channels, height, width)

    # 创建通道注意力模块
    ca = ChannelAttention(in_channels=channels, reduction_ratio=16)

    # 前向传播
    output = ca(x)

    # 打印输入和输出的形状
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    # 检查注意力机制是否改变了特征图的形状
    assert x.shape == output.shape, "输出形状应该与输入形状相同"

    # 可视化注意力权重
    with torch.no_grad():
        # 获取注意力权重
        avg_pool = nn.AdaptiveAvgPool2d(1)(x)
        max_pool = nn.AdaptiveMaxPool2d(1)(x)

        avg_weight = ca.fc(avg_pool)
        max_weight = ca.fc(max_pool)

        attention_weights = ca.sigmoid(avg_weight + max_weight)

        # 打印第一个样本的前10个通道的注意力权重
        print("\n前10个通道的注意力权重:")
        for i in range(min(10, channels)):
            print(f"通道 {i}: {attention_weights[0, i, 0, 0].item():.4f}")

    print("\n通道注意力测试成功!")