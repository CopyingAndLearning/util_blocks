import torch
from diffusers import AutoencoderKL
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class VAETest:
    def __init__(self):
        # Initialize VAE model
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        self.rgb_latent_scale_factor = 0.18215

    def encode_RGB(self, rgb_in: torch.Tensor, generator=None) -> torch.Tensor:
        rgb_latent = self.vae.encode(rgb_in).latent_dist.sample(generator)
        rgb_latent = rgb_latent * self.rgb_latent_scale_factor
        return rgb_latent

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        # Decode latent representation back to image
        latent = latent / self.rgb_latent_scale_factor
        image = self.vae.decode(latent).sample
        return image

def test_vae_encode_decode():
    # Create VAE test instance
    vae_test = VAETest()
    
    # Load and preprocess image
    image = Image.open("../../image/cat.png").convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Encode
    latent = vae_test.encode_RGB(image_tensor)
    
    # Decode
    decoded_image = vae_test.decode_latent(latent)
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    # Show original image
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Show latent space visualization
    plt.subplot(132)
    latent_vis = latent[0, 0].detach().numpy()  # 取第一个通道显示
    plt.imshow(latent_vis, cmap='viridis')
    plt.title('Latent Space')
    plt.colorbar()
    plt.axis('off')
    
    # Show reconstructed image
    plt.subplot(133)
    decoded_image = (decoded_image[0].permute(1, 2, 0).detach().numpy() + 1) / 2
    plt.imshow(decoded_image)
    plt.title('Reconstructed Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_vae_encode_decode()