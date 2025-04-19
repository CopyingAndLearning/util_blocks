import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 20 * 2)  # 输出均值和方差
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(20, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )
        
    def reparameterize(self, mu, logvar): # 参数重整化
        std = torch.exp(0.5 * logvar)     # 生成方差
        eps = torch.randn_like(std)       # 从正态分布里面抽样
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)
        mu = h[:, :20]     # 将encoder输出的内容，拆为均值和方差两个部分；
        logvar = h[:, 20:]
        print(logvar.shape)
        z = self.reparameterize(mu, logvar)    #
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

# 损失函数
def elbo_loss(recon_x, x, mu, logvar):
    # 重建损失（这里使用二元交叉熵）
    recon_loss = nn.BCELoss(reduction='sum')(recon_x, x)
    
    # KL散度
    kl_divergence = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar)
    
    return (recon_loss + kl_divergence) / x.size(0)  # 平均损失

# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = VAE()
    
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