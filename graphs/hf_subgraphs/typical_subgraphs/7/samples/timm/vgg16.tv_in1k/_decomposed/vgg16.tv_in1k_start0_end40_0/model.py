import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32):
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
        tmp_25 = in_25
        tmp_26 = in_26
        tmp_27 = in_27
        tmp_28 = in_28
        tmp_29 = in_29
        tmp_30 = in_30
        tmp_31 = in_31
        tmp_32 = in_32
        tmp_33 = torch.conv2d(tmp_32, tmp_1, tmp_0, (1, 1), (1, 1), (1, 1), 1)
        tmp_32 = tmp_1 = tmp_0 = None
        tmp_34 = torch.nn.functional.relu(tmp_33, inplace=True)
        tmp_33 = None
        tmp_35 = torch.conv2d(tmp_34, tmp_21, tmp_20, (1, 1), (1, 1), (1, 1), 1)
        tmp_34 = tmp_21 = tmp_20 = None
        tmp_36 = torch.nn.functional.relu(tmp_35, inplace=True)
        tmp_35 = None
        tmp_37 = torch.nn.functional.max_pool2d(tmp_36, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_36 = None
        tmp_38 = torch.conv2d(tmp_37, tmp_23, tmp_22, (1, 1), (1, 1), (1, 1), 1)
        tmp_37 = tmp_23 = tmp_22 = None
        tmp_39 = torch.nn.functional.relu(tmp_38, inplace=True)
        tmp_38 = None
        tmp_40 = torch.conv2d(tmp_39, tmp_25, tmp_24, (1, 1), (1, 1), (1, 1), 1)
        tmp_39 = tmp_25 = tmp_24 = None
        tmp_41 = torch.nn.functional.relu(tmp_40, inplace=True)
        tmp_40 = None
        tmp_42 = torch.nn.functional.max_pool2d(tmp_41, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_41 = None
        tmp_43 = torch.conv2d(tmp_42, tmp_3, tmp_2, (1, 1), (1, 1), (1, 1), 1)
        tmp_42 = tmp_3 = tmp_2 = None
        tmp_44 = torch.nn.functional.relu(tmp_43, inplace=True)
        tmp_43 = None
        tmp_45 = torch.conv2d(tmp_44, tmp_5, tmp_4, (1, 1), (1, 1), (1, 1), 1)
        tmp_44 = tmp_5 = tmp_4 = None
        tmp_46 = torch.nn.functional.relu(tmp_45, inplace=True)
        tmp_45 = None
        tmp_47 = torch.conv2d(tmp_46, tmp_7, tmp_6, (1, 1), (1, 1), (1, 1), 1)
        tmp_46 = tmp_7 = tmp_6 = None
        tmp_48 = torch.nn.functional.relu(tmp_47, inplace=True)
        tmp_47 = None
        tmp_49 = torch.nn.functional.max_pool2d(tmp_48, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_48 = None
        tmp_50 = torch.conv2d(tmp_49, tmp_9, tmp_8, (1, 1), (1, 1), (1, 1), 1)
        tmp_49 = tmp_9 = tmp_8 = None
        tmp_51 = torch.nn.functional.relu(tmp_50, inplace=True)
        tmp_50 = None
        tmp_52 = torch.conv2d(tmp_51, tmp_11, tmp_10, (1, 1), (1, 1), (1, 1), 1)
        tmp_51 = tmp_11 = tmp_10 = None
        tmp_53 = torch.nn.functional.relu(tmp_52, inplace=True)
        tmp_52 = None
        tmp_54 = torch.conv2d(tmp_53, tmp_13, tmp_12, (1, 1), (1, 1), (1, 1), 1)
        tmp_53 = tmp_13 = tmp_12 = None
        tmp_55 = torch.nn.functional.relu(tmp_54, inplace=True)
        tmp_54 = None
        tmp_56 = torch.nn.functional.max_pool2d(tmp_55, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_55 = None
        tmp_57 = torch.conv2d(tmp_56, tmp_15, tmp_14, (1, 1), (1, 1), (1, 1), 1)
        tmp_56 = tmp_15 = tmp_14 = None
        tmp_58 = torch.nn.functional.relu(tmp_57, inplace=True)
        tmp_57 = None
        tmp_59 = torch.conv2d(tmp_58, tmp_17, tmp_16, (1, 1), (1, 1), (1, 1), 1)
        tmp_58 = tmp_17 = tmp_16 = None
        tmp_60 = torch.nn.functional.relu(tmp_59, inplace=True)
        tmp_59 = None
        tmp_61 = torch.conv2d(tmp_60, tmp_19, tmp_18, (1, 1), (1, 1), (1, 1), 1)
        tmp_60 = tmp_19 = tmp_18 = None
        tmp_62 = torch.nn.functional.relu(tmp_61, inplace=True)
        tmp_61 = None
        tmp_63 = torch.nn.functional.max_pool2d(tmp_62, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_62 = None
        tmp_64 = torch.conv2d(tmp_63, tmp_29, tmp_28, (1, 1), (0, 0), (1, 1), 1)
        tmp_63 = tmp_29 = tmp_28 = None
        tmp_65 = torch.nn.functional.relu(tmp_64, inplace=True)
        tmp_64 = None
        tmp_66 = torch.nn.functional.dropout(tmp_65, 0.0, False, False)
        tmp_65 = None
        tmp_67 = torch.conv2d(tmp_66, tmp_31, tmp_30, (1, 1), (0, 0), (1, 1), 1)
        tmp_66 = tmp_31 = tmp_30 = None
        tmp_68 = torch.nn.functional.relu(tmp_67, inplace=True)
        tmp_67 = None
        tmp_69 = torch.nn.functional.adaptive_avg_pool2d(tmp_68, 1)
        tmp_68 = None
        tmp_70 = tmp_69.flatten(1, -1)
        tmp_69 = None
        tmp_71 = torch.nn.functional.dropout(tmp_70, 0.0, False, False)
        tmp_70 = None
        tmp_72 = torch.nn.functional.linear(tmp_71, tmp_27, tmp_26)
        tmp_71 = tmp_27 = tmp_26 = None
        return (tmp_72,)