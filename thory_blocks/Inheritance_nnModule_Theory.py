import torch
import torch.nn as nn

class SimpleLinearLayer:
    def __init__(self, input_dim, output_dim):
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim))
        self.bias = nn.Parameter(torch.randn(output_dim))
        # Define activation function
        self.activation = nn.ReLU()

    def forward(self, x):
        # Forward pass: linear transformation + activation function
        x = torch.matmul(x, self.weight.T) + self.bias
        x = self.activation(x)
        return x

    def __call__(self, x):
        # Make instance callable like a function
        return self.forward(x)

# Test code
if __name__ == "__main__":
    # Create a simple linear layer
    layer = SimpleLinearLayer(input_dim=5, output_dim=3)

    # Create input tensor
    input_tensor = torch.randn(1, 5)  # 1 sample, 5 features

    # Call layer
    output_tensor = layer(input_tensor)
    print("Output Tensor:", output_tensor)