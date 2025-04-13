from diffusers import ControlNetModel, DDIMScheduler
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class ControlNetTest:
    def __init__(self):
        self.controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
        self.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.controlnet.to(self.device)

    def test_forward(self, batch_size=1, height=512, width=512):
        # 创建测试输入
        latents = torch.randn(batch_size, 4, height//8, width//8).to(self.device)
        timestep = torch.tensor([1]).to(self.device)
        encoder_hidden_states = torch.randn(batch_size, 77, 768).to(self.device)
        controlnet_cond = torch.randn(batch_size, 3, height, width).to(self.device)

        # ControlNet前向传播
        down_samples, mid_sample = self.controlnet(
            latents,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            return_dict=False,
        )

        return down_samples, mid_sample

def visualize_features(down_samples, mid_sample):
    plt.figure(figsize=(15, 5))
    
    # 显示下采样特征
    plt.subplot(131)
    feature_map = down_samples[0][0, 0].detach().cpu().numpy()
    plt.imshow(feature_map, cmap='viridis')
    plt.title('Down Sample Feature')
    plt.colorbar()
    
    # 显示中间特征
    plt.subplot(132)
    mid_feature = mid_sample[0, 0].detach().cpu().numpy()
    plt.imshow(mid_feature, cmap='viridis')
    plt.title('Mid Block Feature')
    plt.colorbar()
    
    # 显示另一个下采样特征
    plt.subplot(133)
    feature_map_2 = down_samples[-1][0, 0].detach().cpu().numpy()
    plt.imshow(feature_map_2, cmap='viridis')
    plt.title('Last Down Sample Feature')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

def main():
    # 创建测试实例
    test = ControlNetTest()
    
    # 运行测试
    down_samples, mid_sample = test.test_forward()
    
    # 打印特征形状
    print("Down-sampling features shapes:")
    for idx, feature in enumerate(down_samples):
        print(f"Level {idx}: {feature.shape}")
    print(f"Mid block feature shape: {mid_sample.shape}")
    
    # 可视化特征
    visualize_features(down_samples, mid_sample)

if __name__ == "__main__":
    main()