import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, w_26, w_27, w_28, w_29, w_30, w_31, w_32, w_33, w_34, w_35, w_36, w_37, w_38, w_39, w_40, w_41, w_42, w_43, w_44, w_45, w_46, w_47, w_48, w_49, w_50, w_51, w_52, w_53, in_0):
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
        tmp_38 = w_38
        tmp_39 = w_39
        tmp_40 = w_40
        tmp_41 = w_41
        tmp_42 = w_42
        tmp_43 = w_43
        tmp_44 = w_44
        tmp_45 = w_45
        tmp_46 = w_46
        tmp_47 = w_47
        tmp_48 = w_48
        tmp_49 = w_49
        tmp_50 = w_50
        tmp_51 = w_51
        tmp_52 = w_52
        tmp_53 = w_53
        tmp_54 = in_0
        tmp_55 = torch.conv2d(tmp_54, tmp_7, tmp_6, (1, 1), (1, 1), (1, 1), 1)
        tmp_54 = tmp_7 = tmp_6 = None
        tmp_56 = torch.nn.functional.batch_norm(tmp_55, tmp_26, tmp_27, tmp_29, tmp_28, False, 0.1, 1e-05)
        tmp_55 = tmp_26 = tmp_27 = tmp_29 = tmp_28 = None
        tmp_57 = torch.nn.functional.relu(tmp_56, inplace=True)
        tmp_56 = None
        tmp_58 = torch.nn.functional.max_pool2d(tmp_57, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_57 = None
        tmp_59 = torch.conv2d(tmp_58, tmp_43, tmp_42, (1, 1), (1, 1), (1, 1), 1)
        tmp_58 = tmp_43 = tmp_42 = None
        tmp_60 = torch.nn.functional.batch_norm(tmp_59, tmp_44, tmp_45, tmp_47, tmp_46, False, 0.1, 1e-05)
        tmp_59 = tmp_44 = tmp_45 = tmp_47 = tmp_46 = None
        tmp_61 = torch.nn.functional.relu(tmp_60, inplace=True)
        tmp_60 = None
        tmp_62 = torch.nn.functional.max_pool2d(tmp_61, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_61 = None
        tmp_63 = torch.conv2d(tmp_62, tmp_49, tmp_48, (1, 1), (1, 1), (1, 1), 1)
        tmp_62 = tmp_49 = tmp_48 = None
        tmp_64 = torch.nn.functional.batch_norm(tmp_63, tmp_50, tmp_51, tmp_53, tmp_52, False, 0.1, 1e-05)
        tmp_63 = tmp_50 = tmp_51 = tmp_53 = tmp_52 = None
        tmp_65 = torch.nn.functional.relu(tmp_64, inplace=True)
        tmp_64 = None
        tmp_66 = torch.conv2d(tmp_65, tmp_9, tmp_8, (1, 1), (1, 1), (1, 1), 1)
        tmp_65 = tmp_9 = tmp_8 = None
        tmp_67 = torch.nn.functional.batch_norm(tmp_66, tmp_10, tmp_11, tmp_13, tmp_12, False, 0.1, 1e-05)
        tmp_66 = tmp_10 = tmp_11 = tmp_13 = tmp_12 = None
        tmp_68 = torch.nn.functional.relu(tmp_67, inplace=True)
        tmp_67 = None
        tmp_69 = torch.nn.functional.max_pool2d(tmp_68, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_68 = None
        tmp_70 = torch.conv2d(tmp_69, tmp_15, tmp_14, (1, 1), (1, 1), (1, 1), 1)
        tmp_69 = tmp_15 = tmp_14 = None
        tmp_71 = torch.nn.functional.batch_norm(tmp_70, tmp_16, tmp_17, tmp_19, tmp_18, False, 0.1, 1e-05)
        tmp_70 = tmp_16 = tmp_17 = tmp_19 = tmp_18 = None
        tmp_72 = torch.nn.functional.relu(tmp_71, inplace=True)
        tmp_71 = None
        tmp_73 = torch.conv2d(tmp_72, tmp_21, tmp_20, (1, 1), (1, 1), (1, 1), 1)
        tmp_72 = tmp_21 = tmp_20 = None
        tmp_74 = torch.nn.functional.batch_norm(tmp_73, tmp_22, tmp_23, tmp_25, tmp_24, False, 0.1, 1e-05)
        tmp_73 = tmp_22 = tmp_23 = tmp_25 = tmp_24 = None
        tmp_75 = torch.nn.functional.relu(tmp_74, inplace=True)
        tmp_74 = None
        tmp_76 = torch.nn.functional.max_pool2d(tmp_75, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_75 = None
        tmp_77 = torch.conv2d(tmp_76, tmp_31, tmp_30, (1, 1), (1, 1), (1, 1), 1)
        tmp_76 = tmp_31 = tmp_30 = None
        tmp_78 = torch.nn.functional.batch_norm(tmp_77, tmp_32, tmp_33, tmp_35, tmp_34, False, 0.1, 1e-05)
        tmp_77 = tmp_32 = tmp_33 = tmp_35 = tmp_34 = None
        tmp_79 = torch.nn.functional.relu(tmp_78, inplace=True)
        tmp_78 = None
        tmp_80 = torch.conv2d(tmp_79, tmp_37, tmp_36, (1, 1), (1, 1), (1, 1), 1)
        tmp_79 = tmp_37 = tmp_36 = None
        tmp_81 = torch.nn.functional.batch_norm(tmp_80, tmp_38, tmp_39, tmp_41, tmp_40, False, 0.1, 1e-05)
        tmp_80 = tmp_38 = tmp_39 = tmp_41 = tmp_40 = None
        tmp_82 = torch.nn.functional.relu(tmp_81, inplace=True)
        tmp_81 = None
        tmp_83 = torch.nn.functional.max_pool2d(tmp_82, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_82 = None
        tmp_84 = torch.nn.functional.adaptive_avg_pool2d(tmp_83, (7, 7))
        tmp_83 = None
        tmp_85 = torch.flatten(tmp_84, 1)
        tmp_84 = None
        tmp_86 = torch.nn.functional.linear(tmp_85, tmp_1, tmp_0)
        tmp_85 = tmp_1 = tmp_0 = None
        tmp_87 = torch.nn.functional.relu(tmp_86, inplace=True)
        tmp_86 = None
        tmp_88 = torch.nn.functional.dropout(tmp_87, 0.5, False, False)
        tmp_87 = None
        tmp_89 = torch.nn.functional.linear(tmp_88, tmp_3, tmp_2)
        tmp_88 = tmp_3 = tmp_2 = None
        tmp_90 = torch.nn.functional.relu(tmp_89, inplace=True)
        tmp_89 = None
        tmp_91 = torch.nn.functional.dropout(tmp_90, 0.5, False, False)
        tmp_90 = None
        tmp_92 = torch.nn.functional.linear(tmp_91, tmp_5, tmp_4)
        tmp_91 = tmp_5 = tmp_4 = None
        return (tmp_92,)