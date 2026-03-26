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
        tmp_12 = torch.nn.functional.silu(in_12, inplace=True)
        tmp_13 = torch.functional.split(tmp_12, [288, 288, 288, 288], 1)
        tmp_12 = None
        tmp_14 = tmp_13[0]
        tmp_15 = tmp_13[1]
        tmp_16 = tmp_13[2]
        tmp_17 = tmp_13[3]
        tmp_13 = None
        tmp_18 = torch.conv2d(tmp_14, tmp_4, None, (2, 2), (1, 1), (1, 1), 288)
        tmp_14 = tmp_4 = None
        tmp_19 = torch.conv2d(tmp_15, tmp_5, None, (2, 2), (2, 2), (1, 1), 288)
        tmp_15 = tmp_5 = None
        tmp_20 = torch.conv2d(tmp_16, tmp_6, None, (2, 2), (3, 3), (1, 1), 288)
        tmp_16 = tmp_6 = None
        tmp_21 = torch.conv2d(tmp_17, tmp_7, None, (2, 2), (4, 4), (1, 1), 288)
        tmp_17 = tmp_7 = None
        tmp_22 = torch.cat([tmp_18, tmp_19, tmp_20, tmp_21], 1)
        tmp_18 = tmp_19 = tmp_20 = tmp_21 = None
        tmp_23 = torch.nn.functional.batch_norm(tmp_22, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_22 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_24 = torch.nn.functional.silu(tmp_23, inplace=True)
        tmp_23 = None
        tmp_25 = tmp_24.mean((2, 3), keepdim=True)
        tmp_26 = torch.conv2d(tmp_25, tmp_11, tmp_10, (1, 1), (0, 0), (1, 1), 1)
        tmp_25 = tmp_11 = tmp_10 = None
        tmp_27 = torch.nn.functional.silu(tmp_26, inplace=True)
        tmp_26 = None
        tmp_28 = torch.conv2d(tmp_27, tmp_9, tmp_8, (1, 1), (0, 0), (1, 1), 1)
        tmp_27 = tmp_9 = tmp_8 = None
        tmp_29 = torch.sigmoid(tmp_28)
        tmp_28 = None
        tmp_30 = tmp_24 * tmp_29
        tmp_24 = tmp_29 = None
        return (tmp_30,)