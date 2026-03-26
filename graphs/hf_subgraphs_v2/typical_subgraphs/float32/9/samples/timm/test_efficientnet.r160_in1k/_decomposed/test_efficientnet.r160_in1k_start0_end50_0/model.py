import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, w_26, w_27, w_28, w_29, w_30, w_31, w_32, w_33, w_34, w_35, w_36, w_37, w_38, w_39, w_40, w_41, w_42, w_43, w_44, w_45, w_46, w_47, w_48, w_49, w_50, w_51, w_52, w_53, w_54, w_55, w_56, w_57, w_58, w_59, w_60, w_61, w_62, w_63, w_64, w_65, w_66, w_67, w_68, w_69, w_70, w_71, w_72, w_73, w_74, in_0):
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
        tmp_75 = in_0
        tmp_76 = torch.conv2d(tmp_75, tmp_74, None, (2, 2), (1, 1), (1, 1), 1)
        tmp_75 = tmp_74 = None
        tmp_77 = torch.nn.functional.batch_norm(tmp_76, tmp_63, tmp_64, tmp_66, tmp_65, False, 0.1, 1e-05)
        tmp_76 = tmp_63 = tmp_64 = tmp_66 = tmp_65 = None
        tmp_78 = torch.nn.functional.silu(tmp_77, inplace=True)
        tmp_77 = None
        tmp_79 = torch.conv2d(tmp_78, tmp_4, None, (1, 1), (1, 1), (1, 1), 1)
        tmp_78 = tmp_4 = None
        tmp_80 = torch.nn.functional.batch_norm(tmp_79, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_79 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_81 = torch.nn.functional.silu(tmp_80, inplace=True)
        tmp_80 = None
        tmp_82 = torch.conv2d(tmp_81, tmp_13, None, (2, 2), (1, 1), (1, 1), 1)
        tmp_81 = tmp_13 = None
        tmp_83 = torch.nn.functional.batch_norm(tmp_82, tmp_5, tmp_6, tmp_8, tmp_7, False, 0.1, 1e-05)
        tmp_82 = tmp_5 = tmp_6 = tmp_8 = tmp_7 = None
        tmp_84 = torch.nn.functional.silu(tmp_83, inplace=True)
        tmp_83 = None
        tmp_85 = torch.conv2d(tmp_84, tmp_14, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_84 = tmp_14 = None
        tmp_86 = torch.nn.functional.batch_norm(tmp_85, tmp_9, tmp_10, tmp_12, tmp_11, False, 0.1, 1e-05)
        tmp_85 = tmp_9 = tmp_10 = tmp_12 = tmp_11 = None
        tmp_87 = torch.conv2d(tmp_86, tmp_23, None, (2, 2), (1, 1), (1, 1), 1)
        tmp_86 = tmp_23 = None
        tmp_88 = torch.nn.functional.batch_norm(tmp_87, tmp_15, tmp_16, tmp_18, tmp_17, False, 0.1, 1e-05)
        tmp_87 = tmp_15 = tmp_16 = tmp_18 = tmp_17 = None
        tmp_89 = torch.nn.functional.silu(tmp_88, inplace=True)
        tmp_88 = None
        tmp_90 = torch.conv2d(tmp_89, tmp_24, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_89 = tmp_24 = None
        tmp_91 = torch.nn.functional.batch_norm(tmp_90, tmp_19, tmp_20, tmp_22, tmp_21, False, 0.1, 1e-05)
        tmp_90 = tmp_19 = tmp_20 = tmp_22 = tmp_21 = None
        tmp_92 = torch.conv2d(tmp_91, tmp_38, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_91 = tmp_38 = None
        tmp_93 = torch.nn.functional.batch_norm(tmp_92, tmp_25, tmp_26, tmp_28, tmp_27, False, 0.1, 1e-05)
        tmp_92 = tmp_25 = tmp_26 = tmp_28 = tmp_27 = None
        tmp_94 = torch.nn.functional.silu(tmp_93, inplace=True)
        tmp_93 = None
        tmp_95 = torch.conv2d(tmp_94, tmp_37, None, (2, 2), (1, 1), (1, 1), 128)
        tmp_94 = tmp_37 = None
        tmp_96 = torch.nn.functional.batch_norm(tmp_95, tmp_29, tmp_30, tmp_32, tmp_31, False, 0.1, 1e-05)
        tmp_95 = tmp_29 = tmp_30 = tmp_32 = tmp_31 = None
        tmp_97 = torch.nn.functional.silu(tmp_96, inplace=True)
        tmp_96 = None
        tmp_98 = tmp_97.mean((2, 3), keepdim=True)
        tmp_99 = torch.conv2d(tmp_98, tmp_43, tmp_42, (1, 1), (0, 0), (1, 1), 1)
        tmp_98 = tmp_43 = tmp_42 = None
        tmp_100 = torch.nn.functional.silu(tmp_99, inplace=True)
        tmp_99 = None
        tmp_101 = torch.conv2d(tmp_100, tmp_41, tmp_40, (1, 1), (0, 0), (1, 1), 1)
        tmp_100 = tmp_41 = tmp_40 = None
        tmp_102 = torch.sigmoid(tmp_101)
        tmp_101 = None
        tmp_103 = tmp_97 * tmp_102
        tmp_97 = tmp_102 = None
        tmp_104 = torch.conv2d(tmp_103, tmp_39, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_103 = tmp_39 = None
        tmp_105 = torch.nn.functional.batch_norm(tmp_104, tmp_33, tmp_34, tmp_36, tmp_35, False, 0.1, 1e-05)
        tmp_104 = tmp_33 = tmp_34 = tmp_36 = tmp_35 = None
        tmp_106 = torch.conv2d(tmp_105, tmp_57, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_105 = tmp_57 = None
        tmp_107 = torch.nn.functional.batch_norm(tmp_106, tmp_44, tmp_45, tmp_47, tmp_46, False, 0.1, 1e-05)
        tmp_106 = tmp_44 = tmp_45 = tmp_47 = tmp_46 = None
        tmp_108 = torch.nn.functional.silu(tmp_107, inplace=True)
        tmp_107 = None
        tmp_109 = torch.conv2d(tmp_108, tmp_56, None, (2, 2), (1, 1), (1, 1), 192)
        tmp_108 = tmp_56 = None
        tmp_110 = torch.nn.functional.batch_norm(tmp_109, tmp_48, tmp_49, tmp_51, tmp_50, False, 0.1, 1e-05)
        tmp_109 = tmp_48 = tmp_49 = tmp_51 = tmp_50 = None
        tmp_111 = torch.nn.functional.silu(tmp_110, inplace=True)
        tmp_110 = None
        tmp_112 = tmp_111.mean((2, 3), keepdim=True)
        tmp_113 = torch.conv2d(tmp_112, tmp_62, tmp_61, (1, 1), (0, 0), (1, 1), 1)
        tmp_112 = tmp_62 = tmp_61 = None
        tmp_114 = torch.nn.functional.silu(tmp_113, inplace=True)
        tmp_113 = None
        tmp_115 = torch.conv2d(tmp_114, tmp_60, tmp_59, (1, 1), (0, 0), (1, 1), 1)
        tmp_114 = tmp_60 = tmp_59 = None
        tmp_116 = torch.sigmoid(tmp_115)
        tmp_115 = None
        tmp_117 = tmp_111 * tmp_116
        tmp_111 = tmp_116 = None
        tmp_118 = torch.conv2d(tmp_117, tmp_58, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_117 = tmp_58 = None
        tmp_119 = torch.nn.functional.batch_norm(tmp_118, tmp_52, tmp_53, tmp_55, tmp_54, False, 0.1, 1e-05)
        tmp_118 = tmp_52 = tmp_53 = tmp_55 = tmp_54 = None
        tmp_120 = torch.conv2d(tmp_119, tmp_73, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_119 = tmp_73 = None
        tmp_121 = torch.nn.functional.batch_norm(tmp_120, tmp_67, tmp_68, tmp_70, tmp_69, False, 0.1, 1e-05)
        tmp_120 = tmp_67 = tmp_68 = tmp_70 = tmp_69 = None
        tmp_122 = torch.nn.functional.silu(tmp_121, inplace=True)
        tmp_121 = None
        tmp_123 = torch.nn.functional.adaptive_avg_pool2d(tmp_122, 1)
        tmp_122 = None
        tmp_124 = tmp_123.flatten(1, -1)
        tmp_123 = None
        tmp_125 = torch.nn.functional.linear(tmp_124, tmp_72, tmp_71)
        tmp_124 = tmp_72 = tmp_71 = None
        return (tmp_125,)