import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41, in_42, in_43, in_44, in_45, in_46, in_47, in_48, in_49, in_50, in_51, in_52, in_53, in_54, in_55, in_56, in_57, in_58, in_59, in_60, in_61, in_62, in_63, in_64, in_65, in_66, in_67, in_68, in_69, in_70, in_71, in_72, in_73, in_74, in_75, in_76, in_77, in_78, in_79, in_80, in_81, in_82):
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
        tmp_83 = torch.conv2d(tmp_82, tmp_81, None, (1, 1), (1, 1), (1, 1), 1)
        tmp_82 = tmp_81 = None
        tmp_84 = torch.nn.functional.batch_norm(tmp_83, tmp_77, tmp_78, tmp_80, tmp_79, False, 0.1, 1e-05)
        tmp_83 = tmp_77 = tmp_78 = tmp_80 = tmp_79 = None
        tmp_85 = torch.nn.functional.leaky_relu(tmp_84, 0.01, True)
        tmp_84 = None
        tmp_86 = torch.conv2d(tmp_85, tmp_16, None, (2, 2), (1, 1), (1, 1), 1)
        tmp_85 = tmp_16 = None
        tmp_87 = torch.nn.functional.batch_norm(tmp_86, tmp_12, tmp_13, tmp_15, tmp_14, False, 0.1, 1e-05)
        tmp_86 = tmp_12 = tmp_13 = tmp_15 = tmp_14 = None
        tmp_88 = torch.nn.functional.leaky_relu(tmp_87, 0.01, True)
        tmp_87 = None
        tmp_89 = torch.conv2d(tmp_88, tmp_6, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_6 = None
        tmp_90 = torch.nn.functional.batch_norm(tmp_89, tmp_2, tmp_3, tmp_5, tmp_4, False, 0.1, 1e-05)
        tmp_89 = tmp_2 = tmp_3 = tmp_5 = tmp_4 = None
        tmp_91 = torch.nn.functional.leaky_relu(tmp_90, 0.01, True)
        tmp_90 = None
        tmp_92 = torch.conv2d(tmp_91, tmp_11, None, (1, 1), (1, 1), (1, 1), 1)
        tmp_91 = tmp_11 = None
        tmp_93 = torch.nn.functional.batch_norm(tmp_92, tmp_7, tmp_8, tmp_10, tmp_9, False, 0.1, 1e-05)
        tmp_92 = tmp_7 = tmp_8 = tmp_10 = tmp_9 = None
        tmp_94 = torch.nn.functional.leaky_relu(tmp_93, 0.01, True)
        tmp_93 = None
        tmp_95 = tmp_94 + tmp_88
        tmp_94 = tmp_88 = None
        tmp_96 = torch.conv2d(tmp_95, tmp_31, None, (2, 2), (1, 1), (1, 1), 1)
        tmp_95 = tmp_31 = None
        tmp_97 = torch.nn.functional.batch_norm(tmp_96, tmp_27, tmp_28, tmp_30, tmp_29, False, 0.1, 1e-05)
        tmp_96 = tmp_27 = tmp_28 = tmp_30 = tmp_29 = None
        tmp_98 = torch.nn.functional.leaky_relu(tmp_97, 0.01, True)
        tmp_97 = None
        tmp_99 = torch.conv2d(tmp_98, tmp_21, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_21 = None
        tmp_100 = torch.nn.functional.batch_norm(tmp_99, tmp_17, tmp_18, tmp_20, tmp_19, False, 0.1, 1e-05)
        tmp_99 = tmp_17 = tmp_18 = tmp_20 = tmp_19 = None
        tmp_101 = torch.nn.functional.leaky_relu(tmp_100, 0.01, True)
        tmp_100 = None
        tmp_102 = torch.conv2d(tmp_101, tmp_26, None, (1, 1), (1, 1), (1, 1), 1)
        tmp_101 = tmp_26 = None
        tmp_103 = torch.nn.functional.batch_norm(tmp_102, tmp_22, tmp_23, tmp_25, tmp_24, False, 0.1, 1e-05)
        tmp_102 = tmp_22 = tmp_23 = tmp_25 = tmp_24 = None
        tmp_104 = torch.nn.functional.leaky_relu(tmp_103, 0.01, True)
        tmp_103 = None
        tmp_105 = tmp_104 + tmp_98
        tmp_104 = tmp_98 = None
        tmp_106 = torch.conv2d(tmp_105, tmp_46, None, (2, 2), (1, 1), (1, 1), 1)
        tmp_105 = tmp_46 = None
        tmp_107 = torch.nn.functional.batch_norm(tmp_106, tmp_42, tmp_43, tmp_45, tmp_44, False, 0.1, 1e-05)
        tmp_106 = tmp_42 = tmp_43 = tmp_45 = tmp_44 = None
        tmp_108 = torch.nn.functional.leaky_relu(tmp_107, 0.01, True)
        tmp_107 = None
        tmp_109 = torch.conv2d(tmp_108, tmp_36, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_36 = None
        tmp_110 = torch.nn.functional.batch_norm(tmp_109, tmp_32, tmp_33, tmp_35, tmp_34, False, 0.1, 1e-05)
        tmp_109 = tmp_32 = tmp_33 = tmp_35 = tmp_34 = None
        tmp_111 = torch.nn.functional.leaky_relu(tmp_110, 0.01, True)
        tmp_110 = None
        tmp_112 = torch.conv2d(tmp_111, tmp_41, None, (1, 1), (1, 1), (1, 1), 1)
        tmp_111 = tmp_41 = None
        tmp_113 = torch.nn.functional.batch_norm(tmp_112, tmp_37, tmp_38, tmp_40, tmp_39, False, 0.1, 1e-05)
        tmp_112 = tmp_37 = tmp_38 = tmp_40 = tmp_39 = None
        tmp_114 = torch.nn.functional.leaky_relu(tmp_113, 0.01, True)
        tmp_113 = None
        tmp_115 = tmp_114 + tmp_108
        tmp_114 = tmp_108 = None
        tmp_116 = torch.conv2d(tmp_115, tmp_61, None, (2, 2), (1, 1), (1, 1), 1)
        tmp_115 = tmp_61 = None
        tmp_117 = torch.nn.functional.batch_norm(tmp_116, tmp_57, tmp_58, tmp_60, tmp_59, False, 0.1, 1e-05)
        tmp_116 = tmp_57 = tmp_58 = tmp_60 = tmp_59 = None
        tmp_118 = torch.nn.functional.leaky_relu(tmp_117, 0.01, True)
        tmp_117 = None
        tmp_119 = torch.conv2d(tmp_118, tmp_51, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_51 = None
        tmp_120 = torch.nn.functional.batch_norm(tmp_119, tmp_47, tmp_48, tmp_50, tmp_49, False, 0.1, 1e-05)
        tmp_119 = tmp_47 = tmp_48 = tmp_50 = tmp_49 = None
        tmp_121 = torch.nn.functional.leaky_relu(tmp_120, 0.01, True)
        tmp_120 = None
        tmp_122 = torch.conv2d(tmp_121, tmp_56, None, (1, 1), (1, 1), (1, 1), 1)
        tmp_121 = tmp_56 = None
        tmp_123 = torch.nn.functional.batch_norm(tmp_122, tmp_52, tmp_53, tmp_55, tmp_54, False, 0.1, 1e-05)
        tmp_122 = tmp_52 = tmp_53 = tmp_55 = tmp_54 = None
        tmp_124 = torch.nn.functional.leaky_relu(tmp_123, 0.01, True)
        tmp_123 = None
        tmp_125 = tmp_124 + tmp_118
        tmp_124 = tmp_118 = None
        tmp_126 = torch.conv2d(tmp_125, tmp_76, None, (2, 2), (1, 1), (1, 1), 1)
        tmp_125 = tmp_76 = None
        tmp_127 = torch.nn.functional.batch_norm(tmp_126, tmp_72, tmp_73, tmp_75, tmp_74, False, 0.1, 1e-05)
        tmp_126 = tmp_72 = tmp_73 = tmp_75 = tmp_74 = None
        tmp_128 = torch.nn.functional.leaky_relu(tmp_127, 0.01, True)
        tmp_127 = None
        tmp_129 = torch.conv2d(tmp_128, tmp_66, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_66 = None
        tmp_130 = torch.nn.functional.batch_norm(tmp_129, tmp_62, tmp_63, tmp_65, tmp_64, False, 0.1, 1e-05)
        tmp_129 = tmp_62 = tmp_63 = tmp_65 = tmp_64 = None
        tmp_131 = torch.nn.functional.leaky_relu(tmp_130, 0.01, True)
        tmp_130 = None
        tmp_132 = torch.conv2d(tmp_131, tmp_71, None, (1, 1), (1, 1), (1, 1), 1)
        tmp_131 = tmp_71 = None
        tmp_133 = torch.nn.functional.batch_norm(tmp_132, tmp_67, tmp_68, tmp_70, tmp_69, False, 0.1, 1e-05)
        tmp_132 = tmp_67 = tmp_68 = tmp_70 = tmp_69 = None
        tmp_134 = torch.nn.functional.leaky_relu(tmp_133, 0.01, True)
        tmp_133 = None
        tmp_135 = tmp_134 + tmp_128
        tmp_134 = tmp_128 = None
        tmp_136 = torch.nn.functional.adaptive_avg_pool2d(tmp_135, 1)
        tmp_135 = None
        tmp_137 = tmp_136.flatten(1, -1)
        tmp_136 = None
        tmp_138 = torch.nn.functional.dropout(tmp_137, 0.0, False, False)
        tmp_137 = None
        tmp_139 = torch.nn.functional.linear(tmp_138, tmp_1, tmp_0)
        tmp_138 = tmp_1 = tmp_0 = None
        return (tmp_139,)