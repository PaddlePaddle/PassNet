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
        tmp_12 = torch.nn.functional.silu(in_0, inplace=True)
        tmp_13 = torch.functional.split(tmp_12, [180, 180, 180, 180], 1)
        tmp_12 = None
        tmp_14 = tmp_13[0]
        tmp_15 = tmp_13[1]
        tmp_16 = tmp_13[2]
        tmp_17 = tmp_13[3]
        tmp_13 = None
        tmp_18 = torch.conv2d(tmp_14, tmp_4, None, (2, 2), (1, 1), (1, 1), 180)
        tmp_14 = tmp_4 = None
        tmp_19 = torch.conv2d(tmp_15, tmp_5, None, (2, 2), (2, 2), (1, 1), 180)
        tmp_15 = tmp_5 = None
        tmp_20 = torch.conv2d(tmp_16, tmp_6, None, (2, 2), (3, 3), (1, 1), 180)
        tmp_16 = tmp_6 = None
        tmp_21 = torch.conv2d(tmp_17, tmp_7, None, (2, 2), (4, 4), (1, 1), 180)
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