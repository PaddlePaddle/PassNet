import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_0):
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
        tmp_10 = torch.nn.functional.relu(in_0, inplace=True)
        tmp_11 = torch.conv2d(tmp_10, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_10 = tmp_0 = None
        tmp_12 = torch.nn.functional.batch_norm(tmp_11, tmp_5, tmp_6, tmp_8, tmp_7, False, 0.1, 0.001)
        tmp_11 = tmp_5 = tmp_6 = tmp_8 = tmp_7 = None
        tmp_13 = torch.nn.functional.relu(tmp_12, inplace=True)
        tmp_12 = None
        tmp_14 = torch.conv2d(tmp_13, tmp_9, None, (2, 2), (1, 1), (1, 1), 32)
        tmp_13 = tmp_9 = None
        tmp_15 = torch.nn.functional.batch_norm(tmp_14, tmp_1, tmp_2, tmp_4, tmp_3, False, 0.1, 0.001)
        tmp_14 = tmp_1 = tmp_2 = tmp_4 = tmp_3 = None
        return (tmp_15,)