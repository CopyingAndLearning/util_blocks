import torch

data = [
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ],
    [
        [13, 14, 15, 16],
        [17, 18, 19, 20],
        [21, 22, 23, 24]
    ]
]
tensor = torch.tensor(data)
print(tensor.shape)
repeat_tensor = tensor.repeat((1,2,1))     # (2,3,4) -> (2,6,4)
print(repeat_tensor.shape)

# 重复的维度超过张量本身的维度，那么其会在该张量的前面维度进行填充
# The repeated dimensions exceed the tensor's own dimensions, so it will pad the preceding dimensions of the tensor.
repeat_tensor = tensor.repeat((1,1,1,1))   # (2,3,4) -> (1,2,3,4)
print(repeat_tensor.shape)
repeat_tensor = tensor.repeat((1,1,1,3))   # (2,3,4) -> (1,2,3,12)
print(repeat_tensor.shape)
repeat_tensor = tensor.repeat((2,1,1,3))   # (2,3,4) -> (2,2,3,12)
print(repeat_tensor.shape)