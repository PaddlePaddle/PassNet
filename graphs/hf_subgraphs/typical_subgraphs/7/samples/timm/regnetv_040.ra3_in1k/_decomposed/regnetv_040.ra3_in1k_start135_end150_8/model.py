import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16):
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
        tmp_12 = in_12
        tmp_13 = in_13
        tmp_14 = in_14
        tmp_15 = torch.nn.functional.silu(in_16, inplace=True)
        tmp_16 = torch.conv2d(tmp_15, tmp_0, None, (2, 2), (1, 1), (1, 1), 8)
        tmp_15 = tmp_0 = None
        tmp_17 = tmp_16.mean((2, 3), keepdim=True)
        tmp_18 = torch.conv2d(tmp_17, tmp_8, tmp_7, (1, 1), (0, 0), (1, 1), 1)
        tmp_17 = tmp_8 = tmp_7 = None
        tmp_19 = torch.nn.functional.silu(tmp_18, inplace=True)
        tmp_18 = None
        tmp_20 = torch.conv2d(tmp_19, tmp_10, tmp_9, (1, 1), (0, 0), (1, 1), 1)
        tmp_19 = tmp_10 = tmp_9 = None
        tmp_21 = tmp_20.sigmoid()
        tmp_20 = None
        tmp_22 = tmp_16 * tmp_21
        tmp_16 = tmp_21 = None
        tmp_23 = torch.nn.functional.batch_norm(tmp_22, tmp_3, tmp_4, tmp_6, tmp_5, False, 0.1, 1e-05)
        tmp_22 = tmp_3 = tmp_4 = tmp_6 = tmp_5 = None
        tmp_24 = torch.nn.functional.silu(tmp_23, inplace=True)
        tmp_23 = None
        tmp_25 = torch.conv2d(tmp_24, tmp_1, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_24 = tmp_1 = None
        tmp_26 = torch.conv2d(in_15, tmp_2, None, (2, 2), (0, 0), (1, 1), 1)
        tmp_2 = None
        tmp_27 = tmp_25 + tmp_26
        tmp_25 = tmp_26 = None
        tmp_28 = torch.nn.functional.batch_norm(tmp_27, tmp_11, tmp_12, tmp_14, tmp_13, False, 0.1, 1e-05)
        tmp_27 = tmp_11 = tmp_12 = tmp_14 = tmp_13 = None
        tmp_29 = torch.nn.functional.silu(tmp_28, inplace=True)
        tmp_28 = None
        return (tmp_29,)