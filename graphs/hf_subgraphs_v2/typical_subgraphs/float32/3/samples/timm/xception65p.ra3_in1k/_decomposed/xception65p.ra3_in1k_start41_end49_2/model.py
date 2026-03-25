import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = in_7
        tmp_8 = torch.nn.functional.relu(in_9, inplace=True)
        tmp_9 = torch.conv2d(tmp_8, tmp_1, None, (2, 2), (1, 1), (1, 1), 728)
        tmp_8 = tmp_1 = None
        tmp_10 = torch.conv2d(tmp_9, tmp_2, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_9 = tmp_2 = None
        tmp_11 = torch.conv2d(in_8, tmp_0, None, (2, 2), (0, 0), (1, 1), 1)
        tmp_0 = None
        tmp_12 = tmp_10 + tmp_11
        tmp_10 = tmp_11 = None
        tmp_13 = torch.nn.functional.batch_norm(tmp_12, tmp_3, tmp_4, tmp_6, tmp_5, False, 0.1, 0.001)
        tmp_12 = tmp_3 = tmp_4 = tmp_6 = tmp_5 = None
        tmp_14 = torch.nn.functional.relu(tmp_13, inplace=True)
        tmp_13 = None
        tmp_15 = torch.conv2d(tmp_14, tmp_7, None, (1, 1), (1, 1), (1, 1), 728)
        tmp_7 = None
        return (tmp_14, tmp_15)