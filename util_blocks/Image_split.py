import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


def split_image(image_path, patch_size, show_result=True):
    # Load image and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize image to 256x256
        transforms.ToTensor()  # Convert to tensor
    ])
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension


    # Calculate the number of small images that can be split
    num_patches_y = 256 // patch_size
    num_patches_x = 256 // patch_size

    # Initialize an empty list to store the split small images
    patches = []

    # Traverse the image and split according to patch_size
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # Calculate the index range of the current small image
            start_i, start_j = i * patch_size, j * patch_size
            end_i, end_j = start_i + patch_size, start_j + patch_size

            # Slice and add to patches list
            patches.append(image_tensor[0, :, start_i:end_i, start_j:end_j])

    # Convert patches list to tensor
    patches_tensor = torch.stack(patches)

    # Visualize the result after splitting
    if show_result:
        fig, axes = plt.subplots(num_patches_y, num_patches_x, figsize=(15, 15))
        for i in range(num_patches_y):
            for j in range(num_patches_x):
                axes[i, j].imshow(patches_tensor[i * num_patches_x + j].permute(1, 2, 0).numpy(), cmap='gray')
                axes[i, j].axis('off')  # Hide axes
        plt.show()

if __name__ == '__main__':
    image_path = r"img_path"
    # Define the size of each small image
    patch_size = 56
    split_image(image_path, patch_size)