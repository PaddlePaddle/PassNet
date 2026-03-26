import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = in_7
        tmp_8 = in_8
        tmp_9 = in_9
        tmp_10 = in_10
        tmp_11 = in_11
        tmp_12 = torch.nn.functional.relu(in_12, inplace=True)
        tmp_13 = torch.conv2d(tmp_12, tmp_5, tmp_4, (1, 1), (1, 1), (1, 1), 1)
        tmp_5 = tmp_4 = None
        tmp_14 = torch.nn.functional.batch_norm(tmp_13, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_13 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_15 = torch.nn.functional.relu(tmp_14, inplace=True)
        tmp_14 = None
        tmp_16 = torch.conv2d(tmp_15, tmp_11, tmp_10, (1, 1), (2, 2), (2, 2), 1)
        tmp_11 = tmp_10 = None
        tmp_17 = torch.nn.functional.batch_norm(tmp_16, tmp_6, tmp_7, tmp_9, tmp_8, False, 0.1, 1e-05)
        tmp_16 = tmp_6 = tmp_7 = tmp_9 = tmp_8 = None
        return (tmp_17, tmp_12, tmp_15)