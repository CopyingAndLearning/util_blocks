import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib import font_manager

# 设置中文字体，防止乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取两张图片
img1 = cv2.imread('./image/1.png')
img2 = cv2.imread('./image/22.png')

# 确保两张图片尺寸相同
if img1.shape != img2.shape:
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))


# 分离通道并对每个通道进行傅里叶变换
def process_channels(img):
    channels = cv2.split(img)
    fft_channels = []
    for channel in channels:
        # 傅里叶变换
        f = np.fft.fft2(channel)
        # 将零频率分量移到中心
        fshift = np.fft.fftshift(f)
        fft_channels.append(fshift)
    return fft_channels


# 对两张图片进行傅里叶变换
fft_img1 = process_channels(img1)
fft_img2 = process_channels(img2)


# 交换幅度信息
def swap_magnitude(fft1, fft2):
    # 提取幅度和相位
    magnitude1 = np.abs(fft1)
    phase1 = np.angle(fft1)

    magnitude2 = np.abs(fft2)
    phase2 = np.angle(fft2)

    # 交换幅度
    new_fft1 = magnitude2 * np.exp(1j * phase1)
    new_fft2 = magnitude1 * np.exp(1j * phase2)

    return new_fft1, new_fft2


# 对每个通道交换幅度
swapped_fft_img1 = []
swapped_fft_img2 = []

for i in range(3):  # 对RGB三个通道分别处理
    new_fft1, new_fft2 = swap_magnitude(fft_img1[i], fft_img2[i])
    swapped_fft_img1.append(new_fft1)
    swapped_fft_img2.append(new_fft2)


# 逆傅里叶变换
def inverse_fft(fft_channels):
    channels = []
    for fshift in fft_channels:
        # 将零频率分量移回原位
        f_ishift = np.fft.ifftshift(fshift)
        # 逆傅里叶变换
        img_back = np.fft.ifft2(f_ishift)
        # 取实部
        img_back = np.abs(img_back)
        # 归一化到0-255
        img_back = np.clip(img_back, 0, 255).astype(np.uint8)
        channels.append(img_back)
    # 合并通道
    return cv2.merge(channels)


# 对交换幅度后的图像进行逆傅里叶变换
img1_swapped = inverse_fft(swapped_fft_img1)
img2_swapped = inverse_fft(swapped_fft_img2)

# 可视化结果
plt.figure(figsize=(16, 12))

plt.subplot(231)
plt.title('原始图像1')
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(232)
plt.title('原始图像2')
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(233)
plt.title('图像1的幅度谱(取对数)')
magnitude_spectrum = np.log(1 + np.abs(fft_img1[0]))
plt.imshow(magnitude_spectrum, cmap='gray')
plt.axis('off')

plt.subplot(234)
plt.title('图像2的幅度谱(取对数)')
magnitude_spectrum = np.log(1 + np.abs(fft_img2[0]))
plt.imshow(magnitude_spectrum, cmap='gray')
plt.axis('off')

plt.subplot(235)
plt.title('交换幅度后的图像1')
plt.imshow(cv2.cvtColor(img1_swapped, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(236)
plt.title('交换幅度后的图像2')
plt.imshow(cv2.cvtColor(img2_swapped, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.savefig('./image/fourier_swap_result.png', dpi=300)
plt.show()