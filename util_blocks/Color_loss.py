import torch

def color_loss(y_true, y_pred):
    return torch.mean(torch.abs(torch.mean(y_true, dim=[1, 2, 3]) - torch.mean(y_pred, dim=[1, 2, 3])))

if __name__ == '__main__':
    y_true = torch.tensor([
        [
            [
                # width W→
                [1,2,3],  # height H↓
                [2,4,6],
                [4,8,12]
            ],    # channel R
            [
                [4,5,6],
                [8,10,12],
                [19,20,24],
            ],    # channel G
            [
                [7,8,9],
                [14,16,18],
                [28,32,36],
            ],    # channel B
        ]    # channel C
    ],dtype=torch.float32)    # Batch_size B

    y_pred = torch.tensor([
        [
            [
                # width W→
                [2, 3, 4],  # height H↓
                [3, 5, 7],
                [5, 9, 13]
            ],  # channel R
            [
                [5, 6, 7],
                [9, 11, 13],
                [20, 21, 25],
            ],  # channel G
            [
                [8, 9, 10],
                [15, 17, 19],
                [29, 33, 37],
            ],  # channel B
        ]  # channel C
    ], dtype=torch.float32)  # Batch_size B
    print(y_pred.shape)   # BCHW

    # calculate loss
    print(color_loss(y_true, y_pred))   # all +1, so avg_loss is 1
