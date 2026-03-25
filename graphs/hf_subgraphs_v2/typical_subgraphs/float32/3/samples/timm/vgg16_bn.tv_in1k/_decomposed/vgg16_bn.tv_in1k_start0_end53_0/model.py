import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41, in_42, in_43, in_44, in_45, in_46, in_47, in_48, in_49, in_50, in_51, in_52, in_53, in_54, in_55, in_56, in_57, in_58, in_59, in_60, in_61, in_62, in_63, in_64, in_65, in_66, in_67, in_68, in_69, in_70, in_71, in_72, in_73, in_74, in_75, in_76, in_77, in_78, in_79, in_80, in_81, in_82, in_83, in_84):
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
        tmp_33 = in_33
        tmp_34 = in_34
        tmp_35 = in_35
        tmp_36 = in_36
        tmp_37 = in_37
        tmp_38 = in_38
        tmp_39 = in_39
        tmp_40 = in_40
        tmp_41 = in_41
        tmp_42 = in_42
        tmp_43 = in_43
        tmp_44 = in_44
        tmp_45 = in_45
        tmp_46 = in_46
        tmp_47 = in_47
        tmp_48 = in_48
        tmp_49 = in_49
        tmp_50 = in_50
        tmp_51 = in_51
        tmp_52 = in_52
        tmp_53 = in_53
        tmp_54 = in_54
        tmp_55 = in_55
        tmp_56 = in_56
        tmp_57 = in_57
        tmp_58 = in_58
        tmp_59 = in_59
        tmp_60 = in_60
        tmp_61 = in_61
        tmp_62 = in_62
        tmp_63 = in_63
        tmp_64 = in_64
        tmp_65 = in_65
        tmp_66 = in_66
        tmp_67 = in_67
        tmp_68 = in_68
        tmp_69 = in_69
        tmp_70 = in_70
        tmp_71 = in_71
        tmp_72 = in_72
        tmp_73 = in_73
        tmp_74 = in_74
        tmp_75 = in_75
        tmp_76 = in_76
        tmp_77 = in_77
        tmp_78 = in_78
        tmp_79 = in_79
        tmp_80 = in_80
        tmp_81 = in_81
        tmp_82 = in_82
        tmp_83 = in_83
        tmp_84 = in_84
        tmp_85 = torch.conv2d(tmp_84, tmp_1, tmp_0, (1, 1), (1, 1), (1, 1), 1)
        tmp_84 = tmp_1 = tmp_0 = None
        tmp_86 = torch.nn.functional.batch_norm(tmp_85, tmp_20, tmp_21, tmp_23, tmp_22, False, 0.1, 1e-05)
        tmp_85 = tmp_20 = tmp_21 = tmp_23 = tmp_22 = None
        tmp_87 = torch.nn.functional.relu(tmp_86, inplace=True)
        tmp_86 = None
        tmp_88 = torch.conv2d(tmp_87, tmp_61, tmp_60, (1, 1), (1, 1), (1, 1), 1)
        tmp_87 = tmp_61 = tmp_60 = None
        tmp_89 = torch.nn.functional.batch_norm(tmp_88, tmp_68, tmp_69, tmp_71, tmp_70, False, 0.1, 1e-05)
        tmp_88 = tmp_68 = tmp_69 = tmp_71 = tmp_70 = None
        tmp_90 = torch.nn.functional.relu(tmp_89, inplace=True)
        tmp_89 = None
        tmp_91 = torch.nn.functional.max_pool2d(tmp_90, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_90 = None
        tmp_92 = torch.conv2d(tmp_91, tmp_73, tmp_72, (1, 1), (1, 1), (1, 1), 1)
        tmp_91 = tmp_73 = tmp_72 = None
        tmp_93 = torch.nn.functional.batch_norm(tmp_92, tmp_74, tmp_75, tmp_77, tmp_76, False, 0.1, 1e-05)
        tmp_92 = tmp_74 = tmp_75 = tmp_77 = tmp_76 = None
        tmp_94 = torch.nn.functional.relu(tmp_93, inplace=True)
        tmp_93 = None
        tmp_95 = torch.conv2d(tmp_94, tmp_3, tmp_2, (1, 1), (1, 1), (1, 1), 1)
        tmp_94 = tmp_3 = tmp_2 = None
        tmp_96 = torch.nn.functional.batch_norm(tmp_95, tmp_4, tmp_5, tmp_7, tmp_6, False, 0.1, 1e-05)
        tmp_95 = tmp_4 = tmp_5 = tmp_7 = tmp_6 = None
        tmp_97 = torch.nn.functional.relu(tmp_96, inplace=True)
        tmp_96 = None
        tmp_98 = torch.nn.functional.max_pool2d(tmp_97, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_97 = None
        tmp_99 = torch.conv2d(tmp_98, tmp_9, tmp_8, (1, 1), (1, 1), (1, 1), 1)
        tmp_98 = tmp_9 = tmp_8 = None
        tmp_100 = torch.nn.functional.batch_norm(tmp_99, tmp_10, tmp_11, tmp_13, tmp_12, False, 0.1, 1e-05)
        tmp_99 = tmp_10 = tmp_11 = tmp_13 = tmp_12 = None
        tmp_101 = torch.nn.functional.relu(tmp_100, inplace=True)
        tmp_100 = None
        tmp_102 = torch.conv2d(tmp_101, tmp_15, tmp_14, (1, 1), (1, 1), (1, 1), 1)
        tmp_101 = tmp_15 = tmp_14 = None
        tmp_103 = torch.nn.functional.batch_norm(tmp_102, tmp_16, tmp_17, tmp_19, tmp_18, False, 0.1, 1e-05)
        tmp_102 = tmp_16 = tmp_17 = tmp_19 = tmp_18 = None
        tmp_104 = torch.nn.functional.relu(tmp_103, inplace=True)
        tmp_103 = None
        tmp_105 = torch.conv2d(tmp_104, tmp_25, tmp_24, (1, 1), (1, 1), (1, 1), 1)
        tmp_104 = tmp_25 = tmp_24 = None
        tmp_106 = torch.nn.functional.batch_norm(tmp_105, tmp_26, tmp_27, tmp_29, tmp_28, False, 0.1, 1e-05)
        tmp_105 = tmp_26 = tmp_27 = tmp_29 = tmp_28 = None
        tmp_107 = torch.nn.functional.relu(tmp_106, inplace=True)
        tmp_106 = None
        tmp_108 = torch.nn.functional.max_pool2d(tmp_107, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_107 = None
        tmp_109 = torch.conv2d(tmp_108, tmp_31, tmp_30, (1, 1), (1, 1), (1, 1), 1)
        tmp_108 = tmp_31 = tmp_30 = None
        tmp_110 = torch.nn.functional.batch_norm(tmp_109, tmp_32, tmp_33, tmp_35, tmp_34, False, 0.1, 1e-05)
        tmp_109 = tmp_32 = tmp_33 = tmp_35 = tmp_34 = None
        tmp_111 = torch.nn.functional.relu(tmp_110, inplace=True)
        tmp_110 = None
        tmp_112 = torch.conv2d(tmp_111, tmp_37, tmp_36, (1, 1), (1, 1), (1, 1), 1)
        tmp_111 = tmp_37 = tmp_36 = None
        tmp_113 = torch.nn.functional.batch_norm(tmp_112, tmp_38, tmp_39, tmp_41, tmp_40, False, 0.1, 1e-05)
        tmp_112 = tmp_38 = tmp_39 = tmp_41 = tmp_40 = None
        tmp_114 = torch.nn.functional.relu(tmp_113, inplace=True)
        tmp_113 = None
        tmp_115 = torch.conv2d(tmp_114, tmp_43, tmp_42, (1, 1), (1, 1), (1, 1), 1)
        tmp_114 = tmp_43 = tmp_42 = None
        tmp_116 = torch.nn.functional.batch_norm(tmp_115, tmp_44, tmp_45, tmp_47, tmp_46, False, 0.1, 1e-05)
        tmp_115 = tmp_44 = tmp_45 = tmp_47 = tmp_46 = None
        tmp_117 = torch.nn.functional.relu(tmp_116, inplace=True)
        tmp_116 = None
        tmp_118 = torch.nn.functional.max_pool2d(tmp_117, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_117 = None
        tmp_119 = torch.conv2d(tmp_118, tmp_49, tmp_48, (1, 1), (1, 1), (1, 1), 1)
        tmp_118 = tmp_49 = tmp_48 = None
        tmp_120 = torch.nn.functional.batch_norm(tmp_119, tmp_50, tmp_51, tmp_53, tmp_52, False, 0.1, 1e-05)
        tmp_119 = tmp_50 = tmp_51 = tmp_53 = tmp_52 = None
        tmp_121 = torch.nn.functional.relu(tmp_120, inplace=True)
        tmp_120 = None
        tmp_122 = torch.conv2d(tmp_121, tmp_55, tmp_54, (1, 1), (1, 1), (1, 1), 1)
        tmp_121 = tmp_55 = tmp_54 = None
        tmp_123 = torch.nn.functional.batch_norm(tmp_122, tmp_56, tmp_57, tmp_59, tmp_58, False, 0.1, 1e-05)
        tmp_122 = tmp_56 = tmp_57 = tmp_59 = tmp_58 = None
        tmp_124 = torch.nn.functional.relu(tmp_123, inplace=True)
        tmp_123 = None
        tmp_125 = torch.conv2d(tmp_124, tmp_63, tmp_62, (1, 1), (1, 1), (1, 1), 1)
        tmp_124 = tmp_63 = tmp_62 = None
        tmp_126 = torch.nn.functional.batch_norm(tmp_125, tmp_64, tmp_65, tmp_67, tmp_66, False, 0.1, 1e-05)
        tmp_125 = tmp_64 = tmp_65 = tmp_67 = tmp_66 = None
        tmp_127 = torch.nn.functional.relu(tmp_126, inplace=True)
        tmp_126 = None
        tmp_128 = torch.nn.functional.max_pool2d(tmp_127, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_127 = None
        tmp_129 = torch.conv2d(tmp_128, tmp_81, tmp_80, (1, 1), (0, 0), (1, 1), 1)
        tmp_128 = tmp_81 = tmp_80 = None
        tmp_130 = torch.nn.functional.relu(tmp_129, inplace=True)
        tmp_129 = None
        tmp_131 = torch.nn.functional.dropout(tmp_130, 0.0, False, False)
        tmp_130 = None
        tmp_132 = torch.conv2d(tmp_131, tmp_83, tmp_82, (1, 1), (0, 0), (1, 1), 1)
        tmp_131 = tmp_83 = tmp_82 = None
        tmp_133 = torch.nn.functional.relu(tmp_132, inplace=True)
        tmp_132 = None
        tmp_134 = torch.nn.functional.adaptive_avg_pool2d(tmp_133, 1)
        tmp_133 = None
        tmp_135 = tmp_134.flatten(1, -1)
        tmp_134 = None
        tmp_136 = torch.nn.functional.dropout(tmp_135, 0.0, False, False)
        tmp_135 = None
        tmp_137 = torch.nn.functional.linear(tmp_136, tmp_79, tmp_78)
        tmp_136 = tmp_79 = tmp_78 = None
        return (tmp_137,)