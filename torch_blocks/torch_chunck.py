import torch

# 创建一个形状为 (6, 3) 的张量
x = torch.arange(18).reshape(6, 3)
print("Original Tensor:")
print(x)

# 将张量沿着第 0 维度分割成 3 块
chunks = torch.chunk(x, chunks=3, dim=0)
print("\nChunks:")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(chunk)
    print(chunk.shape)