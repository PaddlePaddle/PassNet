import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19):
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
        tmp_17 = in_17
        tmp_18 = torch.nn.functional.silu(in_19, inplace=True)
        tmp_19 = torch.functional.split(tmp_18, [300, 300, 300, 300], 1)
        tmp_18 = None
        tmp_20 = tmp_19[0]
        tmp_21 = tmp_19[1]
        tmp_22 = tmp_19[2]
        tmp_23 = tmp_19[3]
        tmp_19 = None
        tmp_24 = torch.conv2d(tmp_20, tmp_8, None, (1, 1), (1, 1), (1, 1), 300)
        tmp_20 = tmp_8 = None
        tmp_25 = torch.conv2d(tmp_21, tmp_9, None, (1, 1), (2, 2), (1, 1), 300)
        tmp_21 = tmp_9 = None
        tmp_26 = torch.conv2d(tmp_22, tmp_10, None, (1, 1), (3, 3), (1, 1), 300)
        tmp_22 = tmp_10 = None
        tmp_27 = torch.conv2d(tmp_23, tmp_11, None, (1, 1), (4, 4), (1, 1), 300)
        tmp_23 = tmp_11 = None
        tmp_28 = torch.cat([tmp_24, tmp_25, tmp_26, tmp_27], 1)
        tmp_24 = tmp_25 = tmp_26 = tmp_27 = None
        tmp_29 = torch.nn.functional.batch_norm(tmp_28, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_28 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_30 = torch.nn.functional.silu(tmp_29, inplace=True)
        tmp_29 = None
        tmp_31 = tmp_30.mean((2, 3), keepdim=True)
        tmp_32 = torch.conv2d(tmp_31, tmp_17, tmp_16, (1, 1), (0, 0), (1, 1), 1)
        tmp_31 = tmp_17 = tmp_16 = None
        tmp_33 = torch.nn.functional.silu(tmp_32, inplace=True)
        tmp_32 = None
        tmp_34 = torch.conv2d(tmp_33, tmp_15, tmp_14, (1, 1), (0, 0), (1, 1), 1)
        tmp_33 = tmp_15 = tmp_14 = None
        tmp_35 = torch.sigmoid(tmp_34)
        tmp_34 = None
        tmp_36 = tmp_30 * tmp_35
        tmp_30 = tmp_35 = None
        tmp_37 = torch.functional.split(tmp_36, [600, 600], 1)
        tmp_36 = None
        tmp_38 = tmp_37[0]
        tmp_39 = tmp_37[1]
        tmp_37 = None
        tmp_40 = torch.conv2d(tmp_38, tmp_12, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_38 = tmp_12 = None
        tmp_41 = torch.conv2d(tmp_39, tmp_13, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_39 = tmp_13 = None
        tmp_42 = torch.cat([tmp_40, tmp_41], 1)
        tmp_40 = tmp_41 = None
        tmp_43 = torch.nn.functional.batch_norm(tmp_42, tmp_4, tmp_5, tmp_7, tmp_6, False, 0.1, 1e-05)
        tmp_42 = tmp_4 = tmp_5 = tmp_7 = tmp_6 = None
        tmp_44 = tmp_43 + in_18
        tmp_43 = None
        return (tmp_44,)