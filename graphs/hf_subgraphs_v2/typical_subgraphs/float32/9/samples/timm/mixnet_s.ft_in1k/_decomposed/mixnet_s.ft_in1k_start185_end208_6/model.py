import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, in_0, in_1):
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
        tmp_14 = w_14
        tmp_15 = w_15
        tmp_16 = torch.nn.functional.silu(in_1, inplace=True)
        tmp_17 = torch.functional.split(tmp_16, [240, 240], 1)
        tmp_16 = None
        tmp_18 = tmp_17[0]
        tmp_19 = tmp_17[1]
        tmp_17 = None
        tmp_20 = torch.conv2d(tmp_18, tmp_8, None, (1, 1), (1, 1), (1, 1), 240)
        tmp_18 = tmp_8 = None
        tmp_21 = torch.conv2d(tmp_19, tmp_9, None, (1, 1), (2, 2), (1, 1), 240)
        tmp_19 = tmp_9 = None
        tmp_22 = torch.cat([tmp_20, tmp_21], 1)
        tmp_20 = tmp_21 = None
        tmp_23 = torch.nn.functional.batch_norm(tmp_22, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_22 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_24 = torch.nn.functional.silu(tmp_23, inplace=True)
        tmp_23 = None
        tmp_25 = tmp_24.mean((2, 3), keepdim=True)
        tmp_26 = torch.conv2d(tmp_25, tmp_15, tmp_14, (1, 1), (0, 0), (1, 1), 1)
        tmp_25 = tmp_15 = tmp_14 = None
        tmp_27 = torch.nn.functional.silu(tmp_26, inplace=True)
        tmp_26 = None
        tmp_28 = torch.conv2d(tmp_27, tmp_13, tmp_12, (1, 1), (0, 0), (1, 1), 1)
        tmp_27 = tmp_13 = tmp_12 = None
        tmp_29 = torch.sigmoid(tmp_28)
        tmp_28 = None
        tmp_30 = tmp_24 * tmp_29
        tmp_24 = tmp_29 = None
        tmp_31 = torch.functional.split(tmp_30, [240, 240], 1)
        tmp_30 = None
        tmp_32 = tmp_31[0]
        tmp_33 = tmp_31[1]
        tmp_31 = None
        tmp_34 = torch.conv2d(tmp_32, tmp_10, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_32 = tmp_10 = None
        tmp_35 = torch.conv2d(tmp_33, tmp_11, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_33 = tmp_11 = None
        tmp_36 = torch.cat([tmp_34, tmp_35], 1)
        tmp_34 = tmp_35 = None
        tmp_37 = torch.nn.functional.batch_norm(tmp_36, tmp_4, tmp_5, tmp_7, tmp_6, False, 0.1, 1e-05)
        tmp_36 = tmp_4 = tmp_5 = tmp_7 = tmp_6 = None
        tmp_38 = tmp_37 + in_0
        tmp_37 = None
        return (tmp_38,)