import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = w_8
        tmp_9 = w_9
        tmp_10 = w_10
        tmp_11 = w_11
        tmp_12 = torch.nn.functional.relu(in_0, inplace=True)
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