import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, w_26, w_27, w_28, w_29, w_30, w_31, w_32, w_33, w_34, w_35, w_36, w_37, w_38, w_39, w_40, w_41, w_42, w_43, w_44, w_45, w_46, w_47, w_48, w_49, w_50, w_51, w_52, w_53, w_54, w_55, w_56, w_57, w_58, w_59, w_60, w_61, w_62, w_63, w_64, w_65, w_66, w_67, w_68, w_69, w_70, w_71, w_72, w_73, w_74, w_75, w_76, w_77, w_78, w_79, w_80, w_81, w_82, w_83, w_84, w_85, w_86, w_87, w_88, w_89, w_90, w_91, w_92, w_93, w_94):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = w_1
        tmp_3 = w_2
        tmp_4 = w_3
        tmp_5 = w_4
        tmp_6 = w_5
        tmp_7 = w_6
        tmp_8 = w_7
        tmp_9 = w_8
        tmp_10 = w_9
        tmp_11 = w_10
        tmp_12 = w_11
        tmp_13 = w_12
        tmp_14 = w_13
        tmp_15 = w_14
        tmp_16 = w_15
        tmp_17 = w_16
        tmp_18 = w_17
        tmp_19 = w_18
        tmp_20 = w_19
        tmp_21 = w_20
        tmp_22 = w_21
        tmp_23 = w_22
        tmp_24 = w_23
        tmp_25 = w_24
        tmp_26 = w_25
        tmp_27 = w_26
        tmp_28 = w_27
        tmp_29 = w_28
        tmp_30 = w_29
        tmp_31 = w_30
        tmp_32 = w_31
        tmp_33 = w_32
        tmp_34 = w_33
        tmp_35 = w_34
        tmp_36 = w_35
        tmp_37 = w_36
        tmp_38 = w_37
        tmp_39 = w_38
        tmp_40 = w_39
        tmp_41 = w_40
        tmp_42 = w_41
        tmp_43 = w_42
        tmp_44 = w_43
        tmp_45 = w_44
        tmp_46 = w_45
        tmp_47 = w_46
        tmp_48 = w_47
        tmp_49 = w_48
        tmp_50 = w_49
        tmp_51 = w_50
        tmp_52 = w_51
        tmp_53 = w_52
        tmp_54 = w_53
        tmp_55 = w_54
        tmp_56 = w_55
        tmp_57 = w_56
        tmp_58 = w_57
        tmp_59 = w_58
        tmp_60 = w_59
        tmp_61 = w_60
        tmp_62 = w_61
        tmp_63 = w_62
        tmp_64 = w_63
        tmp_65 = w_64
        tmp_66 = w_65
        tmp_67 = w_66
        tmp_68 = w_67
        tmp_69 = w_68
        tmp_70 = w_69
        tmp_71 = w_70
        tmp_72 = w_71
        tmp_73 = w_72
        tmp_74 = w_73
        tmp_75 = w_74
        tmp_76 = w_75
        tmp_77 = w_76
        tmp_78 = w_77
        tmp_79 = w_78
        tmp_80 = w_79
        tmp_81 = w_80
        tmp_82 = w_81
        tmp_83 = w_82
        tmp_84 = w_83
        tmp_85 = w_84
        tmp_86 = w_85
        tmp_87 = w_86
        tmp_88 = w_87
        tmp_89 = w_88
        tmp_90 = w_89
        tmp_91 = w_90
        tmp_92 = w_91
        tmp_93 = w_92
        tmp_94 = w_93
        tmp_95 = w_94
        tmp_96 = torch.conv2d(tmp_0, tmp_6, tmp_5, (1, 1), (1, 1), (1, 1), 1)
        tmp_0 = tmp_6 = tmp_5 = None
        tmp_97 = torch.nn.functional.batch_norm(tmp_96, tmp_1, tmp_2, tmp_4, tmp_3, False, 0.1, 1e-05)
        tmp_96 = tmp_1 = tmp_2 = tmp_4 = tmp_3 = None
        tmp_98 = torch.nn.functional.relu(tmp_97, inplace=True)
        tmp_97 = None
        tmp_99 = torch.conv2d(tmp_98, tmp_48, tmp_47, (1, 1), (1, 1), (1, 1), 1)
        tmp_98 = tmp_48 = tmp_47 = None
        tmp_100 = torch.nn.functional.batch_norm(tmp_99, tmp_43, tmp_44, tmp_46, tmp_45, False, 0.1, 1e-05)
        tmp_99 = tmp_43 = tmp_44 = tmp_46 = tmp_45 = None
        tmp_101 = torch.nn.functional.relu(tmp_100, inplace=True)
        tmp_100 = None
        tmp_102 = torch.nn.functional.max_pool2d(tmp_101, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_101 = None
        tmp_103 = torch.conv2d(tmp_102, tmp_54, tmp_53, (1, 1), (1, 1), (1, 1), 1)
        tmp_102 = tmp_54 = tmp_53 = None
        tmp_104 = torch.nn.functional.batch_norm(tmp_103, tmp_49, tmp_50, tmp_52, tmp_51, False, 0.1, 1e-05)
        tmp_103 = tmp_49 = tmp_50 = tmp_52 = tmp_51 = None
        tmp_105 = torch.nn.functional.relu(tmp_104, inplace=True)
        tmp_104 = None
        tmp_106 = torch.conv2d(tmp_105, tmp_60, tmp_59, (1, 1), (1, 1), (1, 1), 1)
        tmp_105 = tmp_60 = tmp_59 = None
        tmp_107 = torch.nn.functional.batch_norm(tmp_106, tmp_55, tmp_56, tmp_58, tmp_57, False, 0.1, 1e-05)
        tmp_106 = tmp_55 = tmp_56 = tmp_58 = tmp_57 = None
        tmp_108 = torch.nn.functional.relu(tmp_107, inplace=True)
        tmp_107 = None
        tmp_109 = torch.nn.functional.max_pool2d(tmp_108, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_108 = None
        tmp_110 = torch.conv2d(tmp_109, tmp_66, tmp_65, (1, 1), (1, 1), (1, 1), 1)
        tmp_109 = tmp_66 = tmp_65 = None
        tmp_111 = torch.nn.functional.batch_norm(tmp_110, tmp_61, tmp_62, tmp_64, tmp_63, False, 0.1, 1e-05)
        tmp_110 = tmp_61 = tmp_62 = tmp_64 = tmp_63 = None
        tmp_112 = torch.nn.functional.relu(tmp_111, inplace=True)
        tmp_111 = None
        tmp_113 = torch.conv2d(tmp_112, tmp_72, tmp_71, (1, 1), (1, 1), (1, 1), 1)
        tmp_112 = tmp_72 = tmp_71 = None
        tmp_114 = torch.nn.functional.batch_norm(tmp_113, tmp_67, tmp_68, tmp_70, tmp_69, False, 0.1, 1e-05)
        tmp_113 = tmp_67 = tmp_68 = tmp_70 = tmp_69 = None
        tmp_115 = torch.nn.functional.relu(tmp_114, inplace=True)
        tmp_114 = None
        tmp_116 = torch.conv2d(tmp_115, tmp_78, tmp_77, (1, 1), (1, 1), (1, 1), 1)
        tmp_115 = tmp_78 = tmp_77 = None
        tmp_117 = torch.nn.functional.batch_norm(tmp_116, tmp_73, tmp_74, tmp_76, tmp_75, False, 0.1, 1e-05)
        tmp_116 = tmp_73 = tmp_74 = tmp_76 = tmp_75 = None
        tmp_118 = torch.nn.functional.relu(tmp_117, inplace=True)
        tmp_117 = None
        tmp_119 = torch.nn.functional.max_pool2d(tmp_118, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_118 = None
        tmp_120 = torch.conv2d(tmp_119, tmp_12, tmp_11, (1, 1), (1, 1), (1, 1), 1)
        tmp_119 = tmp_12 = tmp_11 = None
        tmp_121 = torch.nn.functional.batch_norm(tmp_120, tmp_7, tmp_8, tmp_10, tmp_9, False, 0.1, 1e-05)
        tmp_120 = tmp_7 = tmp_8 = tmp_10 = tmp_9 = None
        tmp_122 = torch.nn.functional.relu(tmp_121, inplace=True)
        tmp_121 = None
        tmp_123 = torch.conv2d(tmp_122, tmp_18, tmp_17, (1, 1), (1, 1), (1, 1), 1)
        tmp_122 = tmp_18 = tmp_17 = None
        tmp_124 = torch.nn.functional.batch_norm(tmp_123, tmp_13, tmp_14, tmp_16, tmp_15, False, 0.1, 1e-05)
        tmp_123 = tmp_13 = tmp_14 = tmp_16 = tmp_15 = None
        tmp_125 = torch.nn.functional.relu(tmp_124, inplace=True)
        tmp_124 = None
        tmp_126 = torch.conv2d(tmp_125, tmp_24, tmp_23, (1, 1), (1, 1), (1, 1), 1)
        tmp_125 = tmp_24 = tmp_23 = None
        tmp_127 = torch.nn.functional.batch_norm(tmp_126, tmp_19, tmp_20, tmp_22, tmp_21, False, 0.1, 1e-05)
        tmp_126 = tmp_19 = tmp_20 = tmp_22 = tmp_21 = None
        tmp_128 = torch.nn.functional.relu(tmp_127, inplace=True)
        tmp_127 = None
        tmp_129 = torch.nn.functional.max_pool2d(tmp_128, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_128 = None
        tmp_130 = torch.conv2d(tmp_129, tmp_30, tmp_29, (1, 1), (1, 1), (1, 1), 1)
        tmp_129 = tmp_30 = tmp_29 = None
        tmp_131 = torch.nn.functional.batch_norm(tmp_130, tmp_25, tmp_26, tmp_28, tmp_27, False, 0.1, 1e-05)
        tmp_130 = tmp_25 = tmp_26 = tmp_28 = tmp_27 = None
        tmp_132 = torch.nn.functional.relu(tmp_131, inplace=True)
        tmp_131 = None
        tmp_133 = torch.conv2d(tmp_132, tmp_36, tmp_35, (1, 1), (1, 1), (1, 1), 1)
        tmp_132 = tmp_36 = tmp_35 = None
        tmp_134 = torch.nn.functional.batch_norm(tmp_133, tmp_31, tmp_32, tmp_34, tmp_33, False, 0.1, 1e-05)
        tmp_133 = tmp_31 = tmp_32 = tmp_34 = tmp_33 = None
        tmp_135 = torch.nn.functional.relu(tmp_134, inplace=True)
        tmp_134 = None
        tmp_136 = torch.conv2d(tmp_135, tmp_42, tmp_41, (1, 1), (1, 1), (1, 1), 1)
        tmp_135 = tmp_42 = tmp_41 = None
        tmp_137 = torch.nn.functional.batch_norm(tmp_136, tmp_37, tmp_38, tmp_40, tmp_39, False, 0.1, 1e-05)
        tmp_136 = tmp_37 = tmp_38 = tmp_40 = tmp_39 = None
        tmp_138 = torch.nn.functional.relu(tmp_137, inplace=True)
        tmp_137 = None
        tmp_139 = torch.nn.functional.max_pool2d(tmp_138, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_138 = None
        tmp_140 = torch.conv_transpose2d(tmp_139, tmp_79, None, (2, 2), (1, 1), (0, 0), 1, (1, 1))
        tmp_139 = tmp_79 = None
        tmp_141 = torch.nn.functional.batch_norm(tmp_140, tmp_80, tmp_81, tmp_83, tmp_82, False, 0.1, 1e-05)
        tmp_140 = tmp_80 = tmp_81 = tmp_83 = tmp_82 = None
        tmp_142 = torch.nn.functional.relu(tmp_141, inplace=True)
        tmp_141 = None
        tmp_143 = torch.conv_transpose2d(tmp_142, tmp_84, None, (2, 2), (1, 1), (0, 0), 1, (1, 1))
        tmp_142 = tmp_84 = None
        tmp_144 = torch.nn.functional.batch_norm(tmp_143, tmp_85, tmp_86, tmp_88, tmp_87, False, 0.1, 1e-05)
        tmp_143 = tmp_85 = tmp_86 = tmp_88 = tmp_87 = None
        tmp_145 = torch.nn.functional.relu(tmp_144, inplace=True)
        tmp_144 = None
        tmp_146 = torch.conv_transpose2d(tmp_145, tmp_89, None, (2, 2), (1, 1), (0, 0), 1, (1, 1))
        tmp_145 = tmp_89 = None
        tmp_147 = torch.nn.functional.batch_norm(tmp_146, tmp_90, tmp_91, tmp_93, tmp_92, False, 0.1, 1e-05)
        tmp_146 = tmp_90 = tmp_91 = tmp_93 = tmp_92 = None
        tmp_148 = torch.nn.functional.relu(tmp_147, inplace=True)
        tmp_147 = None
        tmp_149 = torch.conv2d(tmp_148, tmp_95, tmp_94, (1, 1), (0, 0), (1, 1), 1)
        tmp_148 = tmp_95 = tmp_94 = None
        return (tmp_149,)