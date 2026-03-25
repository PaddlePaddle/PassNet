import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_0, in_1):
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
        tmp_10 = torch.nn.functional.silu(in_1, inplace=True)
        tmp_11 = tmp_10.mean((2, 3), keepdim=True)
        tmp_12 = torch.conv2d(tmp_11, tmp_3, tmp_2, (1, 1), (0, 0), (1, 1), 1)
        tmp_11 = tmp_3 = tmp_2 = None
        tmp_13 = torch.nn.functional.silu(tmp_12, inplace=True)
        tmp_12 = None
        tmp_14 = torch.conv2d(tmp_13, tmp_5, tmp_4, (1, 1), (0, 0), (1, 1), 1)
        tmp_13 = tmp_5 = tmp_4 = None
        tmp_15 = tmp_14.sigmoid()
        tmp_14 = None
        tmp_16 = tmp_10 * tmp_15
        tmp_10 = tmp_15 = None
        tmp_17 = torch.conv2d(tmp_16, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_16 = tmp_1 = tmp_0 = None
        tmp_18 = tmp_17 + in_0
        tmp_17 = None
        tmp_19 = torch.nn.functional.batch_norm(tmp_18, tmp_6, tmp_7, tmp_9, tmp_8, False, 0.1, 1e-05)
        tmp_6 = tmp_7 = tmp_9 = tmp_8 = None
        return (tmp_18, tmp_19)