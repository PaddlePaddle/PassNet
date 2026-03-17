import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_1):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = w_1
        tmp_3 = w_2
        tmp_4 = w_3
        tmp_5 = w_4
        tmp_6 = w_5
        tmp_7 = w_6
        tmp_8 = w_7
        tmp_9 = w_8
        tmp_10 = w_9
        tmp_11 = torch.nn.functional.relu(in_1, inplace=True)
        tmp_12 = torch.conv2d(tmp_11, tmp_5, None, (1, 1), (1, 1), (1, 1), 1)
        tmp_11 = tmp_5 = None
        tmp_13 = torch.nn.functional.batch_norm(tmp_12, tmp_1, tmp_2, tmp_4, tmp_3, False, 0.1, 1e-05)
        tmp_12 = tmp_1 = tmp_2 = tmp_4 = tmp_3 = None
        tmp_14 = torch.nn.functional.relu(tmp_13, inplace=True)
        tmp_13 = None
        tmp_15 = torch.conv2d(tmp_0, tmp_10, None, (2, 2), (1, 1), (1, 1), 1)
        tmp_0 = tmp_10 = None
        tmp_16 = torch.nn.functional.batch_norm(tmp_15, tmp_6, tmp_7, tmp_9, tmp_8, False, 0.1, 1e-05)
        tmp_15 = tmp_6 = tmp_7 = tmp_9 = tmp_8 = None
        return (tmp_14, tmp_16)