import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41, in_42, in_43, in_44, in_45, in_46, in_47, in_48, in_49, in_50, in_51, in_52, in_53, in_54, in_55, in_56, in_57, in_58, in_59, in_60, in_61, in_62, in_63, in_64, in_65, in_66, in_67, in_68, in_69, in_70, in_71, in_72, in_73, in_74, in_75, in_76, in_77, in_78, in_79, in_80, in_81, in_82, in_83, in_84, in_85, in_86, in_87, in_88, in_89, in_90, in_91, in_92, in_93, in_94, in_95):
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
        tmp_85 = in_85
        tmp_86 = in_86
        tmp_87 = in_87
        tmp_88 = in_88
        tmp_89 = in_89
        tmp_90 = in_90
        tmp_91 = in_91
        tmp_92 = in_92
        tmp_93 = in_93
        tmp_94 = in_94
        tmp_95 = in_95
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