# From https://github.com/carolineec/informative-drawings
# MIT License

import os
import cv2
import torch
import numpy as np

import torch.nn as nn
from einops import rearrange
from torchvision import transforms
from PIL import Image


norm_layer = nn.InstanceNorm2d


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features)
                        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(Generator, self).__init__()

        # Initial convolution block
        model0 = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    norm_layer(64),
                    nn.ReLU(inplace=True) ]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model1 += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
        self.model1 = nn.Sequential(*model1)

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features//2
        for _ in range(2):
            model3 += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [  nn.ReflectionPad2d(3),
                        nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out


class LineartDetector:
    def __init__(self, annotator_ckpts_path):
        self.annotator_ckpts_path = annotator_ckpts_path
        self.model = self.load_model('sk_model.pth')
        self.model_coarse = self.load_model('sk_model2.pth')

    def load_model(self, name):
        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/" + name
        modelpath = os.path.join(self.annotator_ckpts_path, name)
        if not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=self.annotator_ckpts_path)
        model = Generator(3, 1, 3)
        model.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
        model.eval()
        model = model.cuda()
        return model

    def __call__(self, input_image, coarse):
        model = self.model_coarse if coarse else self.model
        assert input_image.ndim == 3
        image = input_image
        with torch.no_grad():
            image = torch.from_numpy(image).float().cuda()
            image = image / 255.0
            image = rearrange(image, 'h w c -> 1 c h w')
            line = model(image)[0][0]

            line = line.cpu().numpy()
            line = (line * 255.0).clip(0, 255).astype(np.uint8)

            return line

class BatchLineartDetector:
    def __init__(self, annotator_ckpts_path):
        self.annotator_ckpts_path = annotator_ckpts_path
        self.model = self.load_model('sk_model.pth')

    def load_model(self, name):
        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/" + name
        modelpath = os.path.join(self.annotator_ckpts_path, name)
        if not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=self.annotator_ckpts_path)    # To consider loading which weight 
        model = Generator(3, 1, 3)
        model.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
        model.eval()
        return model
    
    def to(self, device, dtype):
        self.model.to(device, dtype=dtype)

    def __call__(self, input_image, mean=-1., std=2.):
        model = self.model
        image = input_image
        with torch.no_grad():
            image = (image - mean) / std
            line = model(image)
        line = 1 - line
        return line


if __name__ == '__main__':
    # Create detector instance
    detector = BatchLineartDetector(annotator_ckpts_path="./models/")

    # Load input image
    input_image = Image.open("./cat.png").convert('RGB')  # Convert to RGB
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(input_image).unsqueeze(0)  # Add batch dimension

    # Call detector
    output = detector(input_tensor)

    # Convert output tensor to PIL image and save
    output_image = transforms.ToPILImage()(output[0])  # Get first batch image
    output_path = os.path.join("./outputs", "lineart_output.png")
    os.makedirs("./outputs", exist_ok=True)  # Ensure output directory exists
    output_image.save(output_path)
    print(f"Lineart saved to: {output_path}")