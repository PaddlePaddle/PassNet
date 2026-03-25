import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = torch.nn.functional.relu(in_1, inplace=True)
        tmp_9 = torch.conv2d(tmp_8, tmp_1, None, (2, 2), (1, 1), (1, 1), 1024)
        tmp_8 = tmp_1 = None
        tmp_10 = torch.conv2d(tmp_9, tmp_2, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_9 = tmp_2 = None
        tmp_11 = torch.conv2d(in_0, tmp_0, None, (2, 2), (0, 0), (1, 1), 1)
        tmp_0 = None
        tmp_12 = tmp_10 + tmp_11
        tmp_10 = tmp_11 = None
        tmp_13 = torch.nn.functional.batch_norm(tmp_12, tmp_3, tmp_4, tmp_6, tmp_5, False, 0.1, 0.001)
        tmp_12 = tmp_3 = tmp_4 = tmp_6 = tmp_5 = None
        tmp_14 = torch.nn.functional.relu(tmp_13, inplace=True)
        tmp_13 = None
        tmp_15 = torch.conv2d(tmp_14, tmp_7, None, (1, 1), (1, 1), (1, 1), 1024)
        tmp_14 = tmp_7 = None
        return (tmp_15,)