import torch
import torch.nn as nn
import torch.optim as optim


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # 编码器 - 使用全连接层而不是卷积层，因为输入是展平的向量
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)  # 输出均值和对数方差
        )

        # 解码器 - 同样使用全连接层
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 确保输出在[0,1]范围内
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 编码
        h = self.encoder(x)
        mu, logvar = h[:, :self.latent_dim], h[:, self.latent_dim:]

        # 重参数化采样
        z = self.reparameterize(mu, logvar)

        # 解码
        recon_x = self.decoder(z)

        return recon_x, mu, logvar


# 损失函数
def elbo_loss(recon_x, x, mu, logvar):
    # 重建损失（这里使用二元交叉熵）
    recon_loss = nn.BCELoss(reduction='sum')(recon_x, x)

    # KL散度
    kl_divergence = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1 - logvar)

    return (recon_loss + kl_divergence) / x.size(0)  # 平均损失


# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = VAE(input_dim=784, hidden_dim=400, latent_dim=20)

    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 创建一些随机数据（模拟MNIST数据集）
    batch_size = 64
    x = torch.randn(batch_size, 784)  # 随机生成的输入
    x = torch.sigmoid(x)  # 确保输入在[0, 1]范围内

    # 前向传播
    recon_x, mu, logvar = model(x)

    # 计算损失
    loss = elbo_loss(recon_x, x, mu, logvar)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"损失: {loss.item()}")
    print(f"重建输出形状: {recon_x.shape}")
    print(f"均值形状: {mu.shape}")
    print(f"对数方差形状: {logvar.shape}")