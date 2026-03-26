import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, w_26, w_27, w_28, w_29, w_30, w_31, w_32, w_33, w_34, w_35, w_36, w_37, in_0):
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
        tmp_16 = w_16
        tmp_17 = w_17
        tmp_18 = w_18
        tmp_19 = w_19
        tmp_20 = w_20
        tmp_21 = w_21
        tmp_22 = w_22
        tmp_23 = w_23
        tmp_24 = w_24
        tmp_25 = w_25
        tmp_26 = w_26
        tmp_27 = w_27
        tmp_28 = w_28
        tmp_29 = w_29
        tmp_30 = w_30
        tmp_31 = w_31
        tmp_32 = w_32
        tmp_33 = w_33
        tmp_34 = w_34
        tmp_35 = w_35
        tmp_36 = w_36
        tmp_37 = w_37
        tmp_38 = in_0
        tmp_39 = torch.conv2d(tmp_38, tmp_7, tmp_6, (1, 1), (1, 1), (1, 1), 1)
        tmp_38 = tmp_7 = tmp_6 = None
        tmp_40 = torch.nn.functional.relu(tmp_39, inplace=True)
        tmp_39 = None
        tmp_41 = torch.conv2d(tmp_40, tmp_27, tmp_26, (1, 1), (1, 1), (1, 1), 1)
        tmp_40 = tmp_27 = tmp_26 = None
        tmp_42 = torch.nn.functional.relu(tmp_41, inplace=True)
        tmp_41 = None
        tmp_43 = torch.nn.functional.max_pool2d(tmp_42, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_42 = None
        tmp_44 = torch.conv2d(tmp_43, tmp_35, tmp_34, (1, 1), (1, 1), (1, 1), 1)
        tmp_43 = tmp_35 = tmp_34 = None
        tmp_45 = torch.nn.functional.relu(tmp_44, inplace=True)
        tmp_44 = None
        tmp_46 = torch.conv2d(tmp_45, tmp_37, tmp_36, (1, 1), (1, 1), (1, 1), 1)
        tmp_45 = tmp_37 = tmp_36 = None
        tmp_47 = torch.nn.functional.relu(tmp_46, inplace=True)
        tmp_46 = None
        tmp_48 = torch.nn.functional.max_pool2d(tmp_47, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_47 = None
        tmp_49 = torch.conv2d(tmp_48, tmp_9, tmp_8, (1, 1), (1, 1), (1, 1), 1)
        tmp_48 = tmp_9 = tmp_8 = None
        tmp_50 = torch.nn.functional.relu(tmp_49, inplace=True)
        tmp_49 = None
        tmp_51 = torch.conv2d(tmp_50, tmp_11, tmp_10, (1, 1), (1, 1), (1, 1), 1)
        tmp_50 = tmp_11 = tmp_10 = None
        tmp_52 = torch.nn.functional.relu(tmp_51, inplace=True)
        tmp_51 = None
        tmp_53 = torch.conv2d(tmp_52, tmp_13, tmp_12, (1, 1), (1, 1), (1, 1), 1)
        tmp_52 = tmp_13 = tmp_12 = None
        tmp_54 = torch.nn.functional.relu(tmp_53, inplace=True)
        tmp_53 = None
        tmp_55 = torch.conv2d(tmp_54, tmp_15, tmp_14, (1, 1), (1, 1), (1, 1), 1)
        tmp_54 = tmp_15 = tmp_14 = None
        tmp_56 = torch.nn.functional.relu(tmp_55, inplace=True)
        tmp_55 = None
        tmp_57 = torch.nn.functional.max_pool2d(tmp_56, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_56 = None
        tmp_58 = torch.conv2d(tmp_57, tmp_17, tmp_16, (1, 1), (1, 1), (1, 1), 1)
        tmp_57 = tmp_17 = tmp_16 = None
        tmp_59 = torch.nn.functional.relu(tmp_58, inplace=True)
        tmp_58 = None
        tmp_60 = torch.conv2d(tmp_59, tmp_19, tmp_18, (1, 1), (1, 1), (1, 1), 1)
        tmp_59 = tmp_19 = tmp_18 = None
        tmp_61 = torch.nn.functional.relu(tmp_60, inplace=True)
        tmp_60 = None
        tmp_62 = torch.conv2d(tmp_61, tmp_21, tmp_20, (1, 1), (1, 1), (1, 1), 1)
        tmp_61 = tmp_21 = tmp_20 = None
        tmp_63 = torch.nn.functional.relu(tmp_62, inplace=True)
        tmp_62 = None
        tmp_64 = torch.conv2d(tmp_63, tmp_23, tmp_22, (1, 1), (1, 1), (1, 1), 1)
        tmp_63 = tmp_23 = tmp_22 = None
        tmp_65 = torch.nn.functional.relu(tmp_64, inplace=True)
        tmp_64 = None
        tmp_66 = torch.nn.functional.max_pool2d(tmp_65, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_65 = None
        tmp_67 = torch.conv2d(tmp_66, tmp_25, tmp_24, (1, 1), (1, 1), (1, 1), 1)
        tmp_66 = tmp_25 = tmp_24 = None
        tmp_68 = torch.nn.functional.relu(tmp_67, inplace=True)
        tmp_67 = None
        tmp_69 = torch.conv2d(tmp_68, tmp_29, tmp_28, (1, 1), (1, 1), (1, 1), 1)
        tmp_68 = tmp_29 = tmp_28 = None
        tmp_70 = torch.nn.functional.relu(tmp_69, inplace=True)
        tmp_69 = None
        tmp_71 = torch.conv2d(tmp_70, tmp_31, tmp_30, (1, 1), (1, 1), (1, 1), 1)
        tmp_70 = tmp_31 = tmp_30 = None
        tmp_72 = torch.nn.functional.relu(tmp_71, inplace=True)
        tmp_71 = None
        tmp_73 = torch.conv2d(tmp_72, tmp_33, tmp_32, (1, 1), (1, 1), (1, 1), 1)
        tmp_72 = tmp_33 = tmp_32 = None
        tmp_74 = torch.nn.functional.relu(tmp_73, inplace=True)
        tmp_73 = None
        tmp_75 = torch.nn.functional.max_pool2d(tmp_74, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_74 = None
        tmp_76 = torch.nn.functional.adaptive_avg_pool2d(tmp_75, (7, 7))
        tmp_75 = None
        tmp_77 = torch.flatten(tmp_76, 1)
        tmp_76 = None
        tmp_78 = torch.nn.functional.linear(tmp_77, tmp_1, tmp_0)
        tmp_77 = tmp_1 = tmp_0 = None
        tmp_79 = torch.nn.functional.relu(tmp_78, inplace=True)
        tmp_78 = None
        tmp_80 = torch.nn.functional.dropout(tmp_79, 0.5, False, False)
        tmp_79 = None
        tmp_81 = torch.nn.functional.linear(tmp_80, tmp_3, tmp_2)
        tmp_80 = tmp_3 = tmp_2 = None
        tmp_82 = torch.nn.functional.relu(tmp_81, inplace=True)
        tmp_81 = None
        tmp_83 = torch.nn.functional.dropout(tmp_82, 0.5, False, False)
        tmp_82 = None
        tmp_84 = torch.nn.functional.linear(tmp_83, tmp_5, tmp_4)
        tmp_83 = tmp_5 = tmp_4 = None
        return (tmp_84,)