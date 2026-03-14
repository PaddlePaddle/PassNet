import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17):
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
        tmp_15 = in_15
        tmp_16 = in_16
        tmp_17 = torch.nn.functional.silu(in_17, inplace=True)
        tmp_18 = torch.functional.split(tmp_17, [80, 80, 80], 1)
        tmp_17 = None
        tmp_19 = tmp_18[0]
        tmp_20 = tmp_18[1]
        tmp_21 = tmp_18[2]
        tmp_18 = None
        tmp_22 = torch.conv2d(tmp_19, tmp_8, None, (2, 2), (1, 1), (1, 1), 80)
        tmp_19 = tmp_8 = None
        tmp_23 = torch.conv2d(tmp_20, tmp_9, None, (2, 2), (2, 2), (1, 1), 80)
        tmp_20 = tmp_9 = None
        tmp_24 = torch.conv2d(tmp_21, tmp_10, None, (2, 2), (3, 3), (1, 1), 80)
        tmp_21 = tmp_10 = None
        tmp_25 = torch.cat([tmp_22, tmp_23, tmp_24], 1)
        tmp_22 = tmp_23 = tmp_24 = None
        tmp_26 = torch.nn.functional.batch_norm(tmp_25, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_25 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_27 = torch.nn.functional.silu(tmp_26, inplace=True)
        tmp_26 = None
        tmp_28 = tmp_27.mean((2, 3), keepdim=True)
        tmp_29 = torch.conv2d(tmp_28, tmp_16, tmp_15, (1, 1), (0, 0), (1, 1), 1)
        tmp_28 = tmp_16 = tmp_15 = None
        tmp_30 = torch.nn.functional.silu(tmp_29, inplace=True)
        tmp_29 = None
        tmp_31 = torch.conv2d(tmp_30, tmp_14, tmp_13, (1, 1), (0, 0), (1, 1), 1)
        tmp_30 = tmp_14 = tmp_13 = None
        tmp_32 = torch.sigmoid(tmp_31)
        tmp_31 = None
        tmp_33 = tmp_27 * tmp_32
        tmp_27 = tmp_32 = None
        tmp_34 = torch.functional.split(tmp_33, [120, 120], 1)
        tmp_33 = None
        tmp_35 = tmp_34[0]
        tmp_36 = tmp_34[1]
        tmp_34 = None
        tmp_37 = torch.conv2d(tmp_35, tmp_11, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_35 = tmp_11 = None
        tmp_38 = torch.conv2d(tmp_36, tmp_12, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_36 = tmp_12 = None
        tmp_39 = torch.cat([tmp_37, tmp_38], 1)
        tmp_37 = tmp_38 = None
        tmp_40 = torch.nn.functional.batch_norm(tmp_39, tmp_4, tmp_5, tmp_7, tmp_6, False, 0.1, 1e-05)
        tmp_39 = tmp_4 = tmp_5 = tmp_7 = tmp_6 = None
        return (tmp_40,)