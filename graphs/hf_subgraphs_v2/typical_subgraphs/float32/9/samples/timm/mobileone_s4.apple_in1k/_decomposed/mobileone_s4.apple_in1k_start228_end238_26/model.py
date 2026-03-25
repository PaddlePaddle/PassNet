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
        tmp_8 = 0 + in_1
        tmp_8 += in_0
        tmp_9 = tmp_8
        tmp_8 = None
        tmp_10 = tmp_9.mean((2, 3), keepdim=True)
        tmp_11 = torch.conv2d(tmp_10, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_10 = tmp_1 = tmp_0 = None
        tmp_12 = torch.nn.functional.relu(tmp_11, inplace=True)
        tmp_11 = None
        tmp_13 = torch.conv2d(tmp_12, tmp_3, tmp_2, (1, 1), (0, 0), (1, 1), 1)
        tmp_12 = tmp_3 = tmp_2 = None
        tmp_14 = tmp_13.sigmoid()
        tmp_13 = None
        tmp_15 = tmp_9 * tmp_14
        tmp_9 = tmp_14 = None
        tmp_16 = torch.nn.functional.relu(tmp_15, inplace=True)
        tmp_15 = None
        tmp_17 = torch.nn.functional.batch_norm(tmp_16, tmp_4, tmp_5, tmp_7, tmp_6, False, 0.1, 1e-05)
        tmp_4 = tmp_5 = tmp_7 = tmp_6 = None
        return (tmp_16, tmp_17)