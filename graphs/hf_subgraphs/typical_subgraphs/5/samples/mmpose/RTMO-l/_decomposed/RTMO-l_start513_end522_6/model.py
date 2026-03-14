import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = in_7
        tmp_8 = torch.sigmoid(in_10)
        tmp_9 = in_10 * tmp_8
        tmp_8 = None
        tmp_10 = torch.conv2d(in_9, tmp_3, tmp_2, (1, 1), (0, 0), (1, 1), 1)
        tmp_3 = tmp_2 = None
        tmp_11 = torch.conv2d(tmp_9, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_12 = torch.conv2d(tmp_9, tmp_5, tmp_4, (1, 1), (0, 0), (1, 1), 1)
        tmp_5 = tmp_4 = None
        tmp_13 = torch.conv2d(tmp_9, tmp_7, tmp_6, (1, 1), (0, 0), (1, 1), 1)
        tmp_9 = tmp_7 = tmp_6 = None
        tmp_14 = in_8.split(256, 1)
        tmp_15 = tmp_14[0]
        tmp_16 = tmp_14[1]
        tmp_14 = None
        return (tmp_15, tmp_10, tmp_11, tmp_12, tmp_13, tmp_16)