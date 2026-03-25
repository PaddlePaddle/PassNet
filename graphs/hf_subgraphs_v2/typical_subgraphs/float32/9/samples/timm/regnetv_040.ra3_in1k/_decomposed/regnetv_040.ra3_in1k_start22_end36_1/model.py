import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, in_0, in_1):
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
        tmp_12 = w_12
        tmp_13 = w_13
        tmp_14 = torch.nn.functional.silu(in_1, inplace=True)
        tmp_15 = torch.conv2d(tmp_14, tmp_0, None, (1, 1), (1, 1), (1, 1), 2)
        tmp_14 = tmp_0 = None
        tmp_16 = tmp_15.mean((2, 3), keepdim=True)
        tmp_17 = torch.conv2d(tmp_16, tmp_7, tmp_6, (1, 1), (0, 0), (1, 1), 1)
        tmp_16 = tmp_7 = tmp_6 = None
        tmp_18 = torch.nn.functional.silu(tmp_17, inplace=True)
        tmp_17 = None
        tmp_19 = torch.conv2d(tmp_18, tmp_9, tmp_8, (1, 1), (0, 0), (1, 1), 1)
        tmp_18 = tmp_9 = tmp_8 = None
        tmp_20 = tmp_19.sigmoid()
        tmp_19 = None
        tmp_21 = tmp_15 * tmp_20
        tmp_15 = tmp_20 = None
        tmp_22 = torch.nn.functional.batch_norm(tmp_21, tmp_2, tmp_3, tmp_5, tmp_4, False, 0.1, 1e-05)
        tmp_21 = tmp_2 = tmp_3 = tmp_5 = tmp_4 = None
        tmp_23 = torch.nn.functional.silu(tmp_22, inplace=True)
        tmp_22 = None
        tmp_24 = torch.conv2d(tmp_23, tmp_1, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_23 = tmp_1 = None
        tmp_25 = tmp_24 + in_0
        tmp_24 = None
        tmp_26 = torch.nn.functional.batch_norm(tmp_25, tmp_10, tmp_11, tmp_13, tmp_12, False, 0.1, 1e-05)
        tmp_25 = tmp_10 = tmp_11 = tmp_13 = tmp_12 = None
        tmp_27 = torch.nn.functional.silu(tmp_26, inplace=True)
        tmp_26 = None
        return (tmp_27,)