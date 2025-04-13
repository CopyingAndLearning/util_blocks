import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# 导入之前定义的 Dilated_Resblock
class Dilated_Resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dilated_Resblock, self).__init__()

        sequence = list()
        sequence += [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=1, dilation=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=2, dilation=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=3, dilation=(3, 3)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=2, dilation=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=1, dilation=(1, 1))
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x) + x
        return out


def test_dilated_resblock():
    # 加载测试图像
    image_path = ".image/cat.jpg"  # 请替换为实际图像路径
    image = Image.open(image_path).convert('RGB')

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    input_tensor = transform(image).unsqueeze(0)  # 添加batch维度

    # 创建模型
    model = Dilated_Resblock(in_channels=3, out_channels=64)

    # 如果有预训练权重，加载权重
    # model.load_state_dict(torch.load('path/to/pretrained/weights.pth'))

    # 设置为评估模式
    model.eval()

    # 进行推理
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # 可视化结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # 显示原始图像
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')

    # 显示处理后的图像
    output_image = output_tensor.squeeze(0).permute(1, 2, 0).numpy()
    output_image = np.clip(output_image, 0, 1)
    ax2.imshow(output_image)
    ax2.set_title('Processed Image')
    ax2.axis('off')

    plt.show()


if __name__ == "__main__":
    test_dilated_resblock()