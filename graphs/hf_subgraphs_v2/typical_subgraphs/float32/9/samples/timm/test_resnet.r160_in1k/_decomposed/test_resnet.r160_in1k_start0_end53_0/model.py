import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, w_26, w_27, w_28, w_29, w_30, w_31, w_32, w_33, w_34, w_35, w_36, w_37, w_38, w_39, w_40, w_41, w_42, w_43, w_44, w_45, w_46, w_47, w_48, w_49, w_50, w_51, w_52, w_53, w_54, w_55, w_56, w_57, w_58, w_59, w_60, w_61, w_62, w_63, w_64, w_65, w_66, w_67, w_68, w_69, w_70, w_71, w_72, w_73, w_74, w_75, w_76, in_0):
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
        tmp_66 = w_66
        tmp_67 = w_67
        tmp_68 = w_68
        tmp_69 = w_69
        tmp_70 = w_70
        tmp_71 = w_71
        tmp_72 = w_72
        tmp_73 = w_73
        tmp_74 = w_74
        tmp_75 = w_75
        tmp_76 = w_76
        tmp_77 = in_0
        tmp_78 = torch.conv2d(tmp_77, tmp_4, None, (2, 2), (1, 1), (1, 1), 1)
        tmp_77 = tmp_4 = None
        tmp_79 = torch.nn.functional.batch_norm(tmp_78, tmp_5, tmp_6, tmp_8, tmp_7, False, 0.1, 1e-05)
        tmp_78 = tmp_5 = tmp_6 = tmp_8 = tmp_7 = None
        tmp_80 = torch.nn.functional.relu(tmp_79, inplace=True)
        tmp_79 = None
        tmp_81 = torch.conv2d(tmp_80, tmp_9, None, (1, 1), (1, 1), (1, 1), 1)
        tmp_80 = tmp_9 = None
        tmp_82 = torch.nn.functional.batch_norm(tmp_81, tmp_10, tmp_11, tmp_13, tmp_12, False, 0.1, 1e-05)
        tmp_81 = tmp_10 = tmp_11 = tmp_13 = tmp_12 = None
        tmp_83 = torch.nn.functional.relu(tmp_82, inplace=True)
        tmp_82 = None
        tmp_84 = torch.conv2d(tmp_83, tmp_14, None, (1, 1), (1, 1), (1, 1), 1)
        tmp_83 = tmp_14 = None
        tmp_85 = torch.nn.functional.batch_norm(tmp_84, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_84 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_86 = torch.nn.functional.relu(tmp_85, inplace=True)
        tmp_85 = None
        tmp_87 = torch.nn.functional.max_pool2d(tmp_86, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
        tmp_86 = None
        tmp_88 = torch.conv2d(tmp_87, tmp_25, None, (1, 1), (1, 1), (1, 1), 1)
        tmp_25 = None
        tmp_89 = torch.nn.functional.batch_norm(tmp_88, tmp_17, tmp_18, tmp_20, tmp_19, False, 0.1, 1e-05)
        tmp_88 = tmp_17 = tmp_18 = tmp_20 = tmp_19 = None
        tmp_90 = torch.nn.functional.relu(tmp_89, inplace=True)
        tmp_89 = None
        tmp_91 = torch.conv2d(tmp_90, tmp_26, None, (1, 1), (1, 1), (1, 1), 1)
        tmp_90 = tmp_26 = None
        tmp_92 = torch.nn.functional.batch_norm(tmp_91, tmp_21, tmp_22, tmp_24, tmp_23, False, 0.1, 1e-05)
        tmp_91 = tmp_21 = tmp_22 = tmp_24 = tmp_23 = None
        tmp_92 += tmp_87
        tmp_93 = tmp_92
        tmp_92 = tmp_87 = None
        tmp_94 = torch.nn.functional.relu(tmp_93, inplace=True)
        tmp_93 = None
        tmp_95 = torch.conv2d(tmp_94, tmp_35, None, (2, 2), (1, 1), (1, 1), 1)
        tmp_35 = None
        tmp_96 = torch.nn.functional.batch_norm(tmp_95, tmp_27, tmp_28, tmp_30, tmp_29, False, 0.1, 1e-05)
        tmp_95 = tmp_27 = tmp_28 = tmp_30 = tmp_29 = None
        tmp_97 = torch.nn.functional.relu(tmp_96, inplace=True)
        tmp_96 = None
        tmp_98 = torch.conv2d(tmp_97, tmp_36, None, (1, 1), (1, 1), (1, 1), 1)
        tmp_97 = tmp_36 = None
        tmp_99 = torch.nn.functional.batch_norm(tmp_98, tmp_31, tmp_32, tmp_34, tmp_33, False, 0.1, 1e-05)
        tmp_98 = tmp_31 = tmp_32 = tmp_34 = tmp_33 = None
        tmp_100 = torch.nn.functional.avg_pool2d(tmp_94, 2, 2, 0, True, False, None)
        tmp_94 = None
        tmp_101 = torch.conv2d(tmp_100, tmp_37, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_100 = tmp_37 = None
        tmp_102 = torch.nn.functional.batch_norm(tmp_101, tmp_38, tmp_39, tmp_41, tmp_40, False, 0.1, 1e-05)
        tmp_101 = tmp_38 = tmp_39 = tmp_41 = tmp_40 = None
        tmp_99 += tmp_102
        tmp_103 = tmp_99
        tmp_99 = tmp_102 = None
        tmp_104 = torch.nn.functional.relu(tmp_103, inplace=True)
        tmp_103 = None
        tmp_105 = torch.conv2d(tmp_104, tmp_54, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_54 = None
        tmp_106 = torch.nn.functional.batch_norm(tmp_105, tmp_42, tmp_43, tmp_45, tmp_44, False, 0.1, 1e-05)
        tmp_105 = tmp_42 = tmp_43 = tmp_45 = tmp_44 = None
        tmp_107 = torch.nn.functional.relu(tmp_106, inplace=True)
        tmp_106 = None
        tmp_108 = torch.conv2d(tmp_107, tmp_55, None, (2, 2), (1, 1), (1, 1), 1)
        tmp_107 = tmp_55 = None
        tmp_109 = torch.nn.functional.batch_norm(tmp_108, tmp_46, tmp_47, tmp_49, tmp_48, False, 0.1, 1e-05)
        tmp_108 = tmp_46 = tmp_47 = tmp_49 = tmp_48 = None
        tmp_110 = torch.nn.functional.relu(tmp_109, inplace=True)
        tmp_109 = None
        tmp_111 = torch.conv2d(tmp_110, tmp_56, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_110 = tmp_56 = None
        tmp_112 = torch.nn.functional.batch_norm(tmp_111, tmp_50, tmp_51, tmp_53, tmp_52, False, 0.1, 1e-05)
        tmp_111 = tmp_50 = tmp_51 = tmp_53 = tmp_52 = None
        tmp_113 = torch.nn.functional.avg_pool2d(tmp_104, 2, 2, 0, True, False, None)
        tmp_104 = None
        tmp_114 = torch.conv2d(tmp_113, tmp_57, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_113 = tmp_57 = None
        tmp_115 = torch.nn.functional.batch_norm(tmp_114, tmp_58, tmp_59, tmp_61, tmp_60, False, 0.1, 1e-05)
        tmp_114 = tmp_58 = tmp_59 = tmp_61 = tmp_60 = None
        tmp_112 += tmp_115
        tmp_116 = tmp_112
        tmp_112 = tmp_115 = None
        tmp_117 = torch.nn.functional.relu(tmp_116, inplace=True)
        tmp_116 = None
        tmp_118 = torch.conv2d(tmp_117, tmp_70, None, (2, 2), (1, 1), (1, 1), 1)
        tmp_70 = None
        tmp_119 = torch.nn.functional.batch_norm(tmp_118, tmp_62, tmp_63, tmp_65, tmp_64, False, 0.1, 1e-05)
        tmp_118 = tmp_62 = tmp_63 = tmp_65 = tmp_64 = None
        tmp_120 = torch.nn.functional.relu(tmp_119, inplace=True)
        tmp_119 = None
        tmp_121 = torch.conv2d(tmp_120, tmp_71, None, (1, 1), (1, 1), (1, 1), 1)
        tmp_120 = tmp_71 = None
        tmp_122 = torch.nn.functional.batch_norm(tmp_121, tmp_66, tmp_67, tmp_69, tmp_68, False, 0.1, 1e-05)
        tmp_121 = tmp_66 = tmp_67 = tmp_69 = tmp_68 = None
        tmp_123 = torch.nn.functional.avg_pool2d(tmp_117, 2, 2, 0, True, False, None)
        tmp_117 = None
        tmp_124 = torch.conv2d(tmp_123, tmp_72, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_123 = tmp_72 = None
        tmp_125 = torch.nn.functional.batch_norm(tmp_124, tmp_73, tmp_74, tmp_76, tmp_75, False, 0.1, 1e-05)
        tmp_124 = tmp_73 = tmp_74 = tmp_76 = tmp_75 = None
        tmp_122 += tmp_125
        tmp_126 = tmp_122
        tmp_122 = tmp_125 = None
        tmp_127 = torch.nn.functional.relu(tmp_126, inplace=True)
        tmp_126 = None
        tmp_128 = torch.nn.functional.adaptive_avg_pool2d(tmp_127, 1)
        tmp_127 = None
        tmp_129 = tmp_128.flatten(1, -1)
        tmp_128 = None
        tmp_130 = torch.nn.functional.linear(tmp_129, tmp_16, tmp_15)
        tmp_129 = tmp_16 = tmp_15 = None
        return (tmp_130,)