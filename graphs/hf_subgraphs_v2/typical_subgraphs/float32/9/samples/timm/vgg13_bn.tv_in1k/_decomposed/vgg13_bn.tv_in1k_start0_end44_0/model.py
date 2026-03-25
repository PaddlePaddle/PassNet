import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, w_26, w_27, w_28, w_29, w_30, w_31, w_32, w_33, w_34, w_35, w_36, w_37, w_38, w_39, w_40, w_41, w_42, w_43, w_44, w_45, w_46, w_47, w_48, w_49, w_50, w_51, w_52, w_53, w_54, w_55, w_56, w_57, w_58, w_59, w_60, w_61, w_62, w_63, w_64, w_65, in_0):
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
        tmp_54 = w_54
        tmp_55 = w_55
        tmp_56 = w_56
        tmp_57 = w_57
        tmp_58 = w_58
        tmp_59 = w_59
        tmp_60 = w_60
        tmp_61 = w_61
        tmp_62 = w_62
        tmp_63 = w_63
        tmp_64 = w_64
        tmp_65 = w_65
        tmp_66 = in_0
        tmp_67 = torch.conv2d(tmp_66, tmp_1, tmp_0, (1, 1), (1, 1), (1, 1), 1)
        tmp_66 = tmp_1 = tmp_0 = None
        tmp_68 = torch.nn.functional.batch_norm(tmp_67, tmp_20, tmp_21, tmp_23, tmp_22, False, 0.1, 1e-05)
        tmp_67 = tmp_20 = tmp_21 = tmp_23 = tmp_22 = None
        tmp_69 = torch.nn.functional.relu(tmp_68, inplace=True)
        tmp_68 = None
        tmp_70 = torch.conv2d(tmp_69, tmp_49, tmp_48, (1, 1), (1, 1), (1, 1), 1)
        tmp_69 = tmp_49 = tmp_48 = None
        tmp_71 = torch.nn.functional.batch_norm(tmp_70, tmp_50, tmp_51, tmp_53, tmp_52, False, 0.1, 1e-05)
        tmp_70 = tmp_50 = tmp_51 = tmp_53 = tmp_52 = None
        tmp_72 = torch.nn.functional.relu(tmp_71, inplace=True)
        tmp_71 = None
        tmp_73 = torch.nn.functional.max_pool2d(tmp_72, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_72 = None
        tmp_74 = torch.conv2d(tmp_73, tmp_55, tmp_54, (1, 1), (1, 1), (1, 1), 1)
        tmp_73 = tmp_55 = tmp_54 = None
        tmp_75 = torch.nn.functional.batch_norm(tmp_74, tmp_56, tmp_57, tmp_59, tmp_58, False, 0.1, 1e-05)
        tmp_74 = tmp_56 = tmp_57 = tmp_59 = tmp_58 = None
        tmp_76 = torch.nn.functional.relu(tmp_75, inplace=True)
        tmp_75 = None
        tmp_77 = torch.conv2d(tmp_76, tmp_3, tmp_2, (1, 1), (1, 1), (1, 1), 1)
        tmp_76 = tmp_3 = tmp_2 = None
        tmp_78 = torch.nn.functional.batch_norm(tmp_77, tmp_4, tmp_5, tmp_7, tmp_6, False, 0.1, 1e-05)
        tmp_77 = tmp_4 = tmp_5 = tmp_7 = tmp_6 = None
        tmp_79 = torch.nn.functional.relu(tmp_78, inplace=True)
        tmp_78 = None
        tmp_80 = torch.nn.functional.max_pool2d(tmp_79, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_79 = None
        tmp_81 = torch.conv2d(tmp_80, tmp_9, tmp_8, (1, 1), (1, 1), (1, 1), 1)
        tmp_80 = tmp_9 = tmp_8 = None
        tmp_82 = torch.nn.functional.batch_norm(tmp_81, tmp_10, tmp_11, tmp_13, tmp_12, False, 0.1, 1e-05)
        tmp_81 = tmp_10 = tmp_11 = tmp_13 = tmp_12 = None
        tmp_83 = torch.nn.functional.relu(tmp_82, inplace=True)
        tmp_82 = None
        tmp_84 = torch.conv2d(tmp_83, tmp_15, tmp_14, (1, 1), (1, 1), (1, 1), 1)
        tmp_83 = tmp_15 = tmp_14 = None
        tmp_85 = torch.nn.functional.batch_norm(tmp_84, tmp_16, tmp_17, tmp_19, tmp_18, False, 0.1, 1e-05)
        tmp_84 = tmp_16 = tmp_17 = tmp_19 = tmp_18 = None
        tmp_86 = torch.nn.functional.relu(tmp_85, inplace=True)
        tmp_85 = None
        tmp_87 = torch.nn.functional.max_pool2d(tmp_86, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_86 = None
        tmp_88 = torch.conv2d(tmp_87, tmp_25, tmp_24, (1, 1), (1, 1), (1, 1), 1)
        tmp_87 = tmp_25 = tmp_24 = None
        tmp_89 = torch.nn.functional.batch_norm(tmp_88, tmp_26, tmp_27, tmp_29, tmp_28, False, 0.1, 1e-05)
        tmp_88 = tmp_26 = tmp_27 = tmp_29 = tmp_28 = None
        tmp_90 = torch.nn.functional.relu(tmp_89, inplace=True)
        tmp_89 = None
        tmp_91 = torch.conv2d(tmp_90, tmp_31, tmp_30, (1, 1), (1, 1), (1, 1), 1)
        tmp_90 = tmp_31 = tmp_30 = None
        tmp_92 = torch.nn.functional.batch_norm(tmp_91, tmp_32, tmp_33, tmp_35, tmp_34, False, 0.1, 1e-05)
        tmp_91 = tmp_32 = tmp_33 = tmp_35 = tmp_34 = None
        tmp_93 = torch.nn.functional.relu(tmp_92, inplace=True)
        tmp_92 = None
        tmp_94 = torch.nn.functional.max_pool2d(tmp_93, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_93 = None
        tmp_95 = torch.conv2d(tmp_94, tmp_37, tmp_36, (1, 1), (1, 1), (1, 1), 1)
        tmp_94 = tmp_37 = tmp_36 = None
        tmp_96 = torch.nn.functional.batch_norm(tmp_95, tmp_38, tmp_39, tmp_41, tmp_40, False, 0.1, 1e-05)
        tmp_95 = tmp_38 = tmp_39 = tmp_41 = tmp_40 = None
        tmp_97 = torch.nn.functional.relu(tmp_96, inplace=True)
        tmp_96 = None
        tmp_98 = torch.conv2d(tmp_97, tmp_43, tmp_42, (1, 1), (1, 1), (1, 1), 1)
        tmp_97 = tmp_43 = tmp_42 = None
        tmp_99 = torch.nn.functional.batch_norm(tmp_98, tmp_44, tmp_45, tmp_47, tmp_46, False, 0.1, 1e-05)
        tmp_98 = tmp_44 = tmp_45 = tmp_47 = tmp_46 = None
        tmp_100 = torch.nn.functional.relu(tmp_99, inplace=True)
        tmp_99 = None
        tmp_101 = torch.nn.functional.max_pool2d(tmp_100, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_100 = None
        tmp_102 = torch.conv2d(tmp_101, tmp_63, tmp_62, (1, 1), (0, 0), (1, 1), 1)
        tmp_101 = tmp_63 = tmp_62 = None
        tmp_103 = torch.nn.functional.relu(tmp_102, inplace=True)
        tmp_102 = None
        tmp_104 = torch.nn.functional.dropout(tmp_103, 0.0, False, False)
        tmp_103 = None
        tmp_105 = torch.conv2d(tmp_104, tmp_65, tmp_64, (1, 1), (0, 0), (1, 1), 1)
        tmp_104 = tmp_65 = tmp_64 = None
        tmp_106 = torch.nn.functional.relu(tmp_105, inplace=True)
        tmp_105 = None
        tmp_107 = torch.nn.functional.adaptive_avg_pool2d(tmp_106, 1)
        tmp_106 = None
        tmp_108 = tmp_107.flatten(1, -1)
        tmp_107 = None
        tmp_109 = torch.nn.functional.dropout(tmp_108, 0.0, False, False)
        tmp_108 = None
        tmp_110 = torch.nn.functional.linear(tmp_109, tmp_61, tmp_60)
        tmp_109 = tmp_61 = tmp_60 = None
        return (tmp_110,)