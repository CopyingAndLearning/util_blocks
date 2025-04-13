import torch
import torch.nn as nn


class Depth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


if __name__ == '__main__':
    # Set input parameters
    batch_size = 2
    in_channels = 64
    out_channels = 128
    height = 32
    width = 32

    # Create model instance
    model = Depth_conv(in_channels, out_channels)

    # Create random input data
    x = torch.randn(batch_size, in_channels, height, width)

    # Forward propagation
    output = model(x)

    # Print input and output shapes
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Verify if the output tensor contains valid values
    print(f"Output contains valid values: {torch.isfinite(output).all()}")