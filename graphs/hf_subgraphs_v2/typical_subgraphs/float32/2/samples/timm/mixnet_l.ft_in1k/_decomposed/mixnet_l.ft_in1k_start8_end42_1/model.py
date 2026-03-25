import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26):
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
        tmp_18 = in_18
        tmp_19 = in_19
        tmp_20 = in_20
        tmp_21 = in_21
        tmp_22 = in_22
        tmp_23 = in_23
        tmp_24 = in_24
        tmp_25 = in_26 + in_25
        tmp_26 = torch.functional.split(tmp_25, [16, 16], 1)
        tmp_25 = None
        tmp_27 = tmp_26[0]
        tmp_28 = tmp_26[1]
        tmp_26 = None
        tmp_29 = torch.conv2d(tmp_27, tmp_15, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_27 = tmp_15 = None
        tmp_30 = torch.conv2d(tmp_28, tmp_16, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_28 = tmp_16 = None
        tmp_31 = torch.cat([tmp_29, tmp_30], 1)
        tmp_29 = tmp_30 = None
        tmp_32 = torch.nn.functional.batch_norm(tmp_31, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_31 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_33 = torch.nn.functional.relu(tmp_32, inplace=True)
        tmp_32 = None
        tmp_34 = torch.functional.split(tmp_33, [64, 64, 64], 1)
        tmp_33 = None
        tmp_35 = tmp_34[0]
        tmp_36 = tmp_34[1]
        tmp_37 = tmp_34[2]
        tmp_34 = None
        tmp_38 = torch.conv2d(tmp_35, tmp_12, None, (2, 2), (1, 1), (1, 1), 64)
        tmp_35 = tmp_12 = None
        tmp_39 = torch.conv2d(tmp_36, tmp_13, None, (2, 2), (2, 2), (1, 1), 64)
        tmp_36 = tmp_13 = None
        tmp_40 = torch.conv2d(tmp_37, tmp_14, None, (2, 2), (3, 3), (1, 1), 64)
        tmp_37 = tmp_14 = None
        tmp_41 = torch.cat([tmp_38, tmp_39, tmp_40], 1)
        tmp_38 = tmp_39 = tmp_40 = None
        tmp_42 = torch.nn.functional.batch_norm(tmp_41, tmp_4, tmp_5, tmp_7, tmp_6, False, 0.1, 1e-05)
        tmp_41 = tmp_4 = tmp_5 = tmp_7 = tmp_6 = None
        tmp_43 = torch.nn.functional.relu(tmp_42, inplace=True)
        tmp_42 = None
        tmp_44 = torch.functional.split(tmp_43, [96, 96], 1)
        tmp_43 = None
        tmp_45 = tmp_44[0]
        tmp_46 = tmp_44[1]
        tmp_44 = None
        tmp_47 = torch.conv2d(tmp_45, tmp_17, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_45 = tmp_17 = None
        tmp_48 = torch.conv2d(tmp_46, tmp_18, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_46 = tmp_18 = None
        tmp_49 = torch.cat([tmp_47, tmp_48], 1)
        tmp_47 = tmp_48 = None
        tmp_50 = torch.nn.functional.batch_norm(tmp_49, tmp_8, tmp_9, tmp_11, tmp_10, False, 0.1, 1e-05)
        tmp_49 = tmp_8 = tmp_9 = tmp_11 = tmp_10 = None
        tmp_51 = torch.functional.split(tmp_50, [20, 20], 1)
        tmp_52 = tmp_51[0]
        tmp_53 = tmp_51[1]
        tmp_51 = None
        tmp_54 = torch.conv2d(tmp_52, tmp_23, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_52 = tmp_23 = None
        tmp_55 = torch.conv2d(tmp_53, tmp_24, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_53 = tmp_24 = None
        tmp_56 = torch.cat([tmp_54, tmp_55], 1)
        tmp_54 = tmp_55 = None
        tmp_57 = torch.nn.functional.batch_norm(tmp_56, tmp_19, tmp_20, tmp_22, tmp_21, False, 0.1, 1e-05)
        tmp_56 = tmp_19 = tmp_20 = tmp_22 = tmp_21 = None
        tmp_58 = torch.nn.functional.relu(tmp_57, inplace=True)
        tmp_57 = None
        return (tmp_50, tmp_58)