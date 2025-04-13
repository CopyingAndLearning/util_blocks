import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


def normalize_and_show_image(image_path, mean=0.5, std=0.5):
    # Load image
    image = Image.open(image_path).convert('RGB')

    # Display original image
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    # Convert to tensor
    transform = transforms.ToTensor()
    image_tensor = transform(image)

    # Normalize image
    with torch.no_grad():
        normalized_image = (image_tensor - mean) / std

    # Convert normalized tensor back to image format for display
    normalized_display = ((normalized_image + 1) / 2).clamp(0,1)  # [-1,1] -> [0,1]
    # normalized_display = normalized_image
    normalized_display = normalized_display.clamp(0, 1)

    # Display normalized image
    plt.subplot(132)
    plt.imshow(normalized_display.permute(1, 2, 0))
    plt.title('Normalized Image')
    plt.axis('off')

    # Display histogram of pixel values
    plt.subplot(133)
    plt.hist(normalized_image.numpy().flatten(), bins=50)
    plt.title('Pixel Value Distribution After Normalization')

    plt.tight_layout()
    plt.show()


# Usage example
if __name__ == "__main__":
    image_path = "./image/cat.jpg"
    normalize_and_show_image(image_path)