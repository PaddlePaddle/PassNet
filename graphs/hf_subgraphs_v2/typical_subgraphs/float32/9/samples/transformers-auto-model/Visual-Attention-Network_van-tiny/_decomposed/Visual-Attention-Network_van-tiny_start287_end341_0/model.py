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
        tmp_54 = torch.nn.functional.batch_norm(in_0, tmp_22, tmp_23, tmp_25, tmp_24, False, 0.1, 1e-05)
        tmp_22 = tmp_23 = tmp_25 = tmp_24 = None
        tmp_55 = torch.conv2d(tmp_54, tmp_9, tmp_8, (1, 1), (0, 0), (1, 1), 1)
        tmp_9 = tmp_8 = None
        tmp_56 = torch.nn.functional.gelu(tmp_55)
        tmp_55 = None
        tmp_57 = torch.conv2d(tmp_56, tmp_3, tmp_2, (1, 1), (2, 2), (1, 1), 256)
        tmp_3 = tmp_2 = None
        tmp_58 = torch.conv2d(tmp_57, tmp_1, tmp_0, (1, 1), (9, 9), (3, 3), 256)
        tmp_57 = tmp_1 = tmp_0 = None
        tmp_59 = torch.conv2d(tmp_58, tmp_5, tmp_4, (1, 1), (0, 0), (1, 1), 1)
        tmp_58 = tmp_5 = tmp_4 = None
        tmp_60 = tmp_56 * tmp_59
        tmp_56 = tmp_59 = None
        tmp_61 = torch.conv2d(tmp_60, tmp_7, tmp_6, (1, 1), (0, 0), (1, 1), 1)
        tmp_60 = tmp_7 = tmp_6 = None
        tmp_62 = tmp_61 + tmp_54
        tmp_61 = tmp_54 = None
        tmp_63 = tmp_10.unsqueeze(-1)
        tmp_10 = None
        tmp_64 = tmp_63.unsqueeze(-1)
        tmp_63 = None
        tmp_65 = tmp_64 * tmp_62
        tmp_64 = tmp_62 = None
        tmp_66 = in_0 + tmp_65
        tmp_65 = None
        tmp_67 = torch.nn.functional.batch_norm(tmp_66, tmp_18, tmp_19, tmp_21, tmp_20, False, 0.1, 1e-05)
        tmp_18 = tmp_19 = tmp_21 = tmp_20 = None
        tmp_68 = torch.conv2d(tmp_67, tmp_14, tmp_13, (1, 1), (0, 0), (1, 1), 1)
        tmp_67 = tmp_14 = tmp_13 = None
        tmp_69 = torch.conv2d(tmp_68, tmp_12, tmp_11, (1, 1), (1, 1), (1, 1), 1024)
        tmp_68 = tmp_12 = tmp_11 = None
        tmp_70 = torch.nn.functional.gelu(tmp_69)
        tmp_69 = None
        tmp_71 = torch.nn.functional.dropout(tmp_70, 0.0, False, False)
        tmp_70 = None
        tmp_72 = torch.conv2d(tmp_71, tmp_16, tmp_15, (1, 1), (0, 0), (1, 1), 1)
        tmp_71 = tmp_16 = tmp_15 = None
        tmp_73 = torch.nn.functional.dropout(tmp_72, 0.0, False, False)
        tmp_72 = None
        tmp_74 = tmp_17.unsqueeze(-1)
        tmp_17 = None
        tmp_75 = tmp_74.unsqueeze(-1)
        tmp_74 = None
        tmp_76 = tmp_75 * tmp_73
        tmp_75 = tmp_73 = None
        tmp_77 = tmp_66 + tmp_76
        tmp_66 = tmp_76 = None
        tmp_78 = torch.nn.functional.batch_norm(tmp_77, tmp_48, tmp_49, tmp_51, tmp_50, False, 0.1, 1e-05)
        tmp_48 = tmp_49 = tmp_51 = tmp_50 = None
        tmp_79 = torch.conv2d(tmp_78, tmp_35, tmp_34, (1, 1), (0, 0), (1, 1), 1)
        tmp_35 = tmp_34 = None
        tmp_80 = torch.nn.functional.gelu(tmp_79)
        tmp_79 = None
        tmp_81 = torch.conv2d(tmp_80, tmp_29, tmp_28, (1, 1), (2, 2), (1, 1), 256)
        tmp_29 = tmp_28 = None
        tmp_82 = torch.conv2d(tmp_81, tmp_27, tmp_26, (1, 1), (9, 9), (3, 3), 256)
        tmp_81 = tmp_27 = tmp_26 = None
        tmp_83 = torch.conv2d(tmp_82, tmp_31, tmp_30, (1, 1), (0, 0), (1, 1), 1)
        tmp_82 = tmp_31 = tmp_30 = None
        tmp_84 = tmp_80 * tmp_83
        tmp_80 = tmp_83 = None
        tmp_85 = torch.conv2d(tmp_84, tmp_33, tmp_32, (1, 1), (0, 0), (1, 1), 1)
        tmp_84 = tmp_33 = tmp_32 = None
        tmp_86 = tmp_85 + tmp_78
        tmp_85 = tmp_78 = None
        tmp_87 = tmp_36.unsqueeze(-1)
        tmp_36 = None
        tmp_88 = tmp_87.unsqueeze(-1)
        tmp_87 = None
        tmp_89 = tmp_88 * tmp_86
        tmp_88 = tmp_86 = None
        tmp_90 = tmp_77 + tmp_89
        tmp_77 = tmp_89 = None
        tmp_91 = torch.nn.functional.batch_norm(tmp_90, tmp_44, tmp_45, tmp_47, tmp_46, False, 0.1, 1e-05)
        tmp_44 = tmp_45 = tmp_47 = tmp_46 = None
        tmp_92 = torch.conv2d(tmp_91, tmp_40, tmp_39, (1, 1), (0, 0), (1, 1), 1)
        tmp_91 = tmp_40 = tmp_39 = None
        tmp_93 = torch.conv2d(tmp_92, tmp_38, tmp_37, (1, 1), (1, 1), (1, 1), 1024)
        tmp_92 = tmp_38 = tmp_37 = None
        tmp_94 = torch.nn.functional.gelu(tmp_93)
        tmp_93 = None
        tmp_95 = torch.nn.functional.dropout(tmp_94, 0.0, False, False)
        tmp_94 = None
        tmp_96 = torch.conv2d(tmp_95, tmp_42, tmp_41, (1, 1), (0, 0), (1, 1), 1)
        tmp_95 = tmp_42 = tmp_41 = None
        tmp_97 = torch.nn.functional.dropout(tmp_96, 0.0, False, False)
        tmp_96 = None
        tmp_98 = tmp_43.unsqueeze(-1)
        tmp_43 = None
        tmp_99 = tmp_98.unsqueeze(-1)
        tmp_98 = None
        tmp_100 = tmp_99 * tmp_97
        tmp_99 = tmp_97 = None
        tmp_101 = tmp_90 + tmp_100
        tmp_90 = tmp_100 = None
        tmp_102 = tmp_101.flatten(2)
        tmp_101 = None
        tmp_103 = tmp_102.transpose(1, 2)
        tmp_102 = None
        tmp_104 = torch.nn.functional.layer_norm(tmp_103, (256,), tmp_53, tmp_52, 1e-06)
        tmp_103 = tmp_53 = tmp_52 = None
        tmp_105 = tmp_104.view(1, 7, 7, 256)
        tmp_104 = None
        tmp_106 = tmp_105.permute(0, 3, 1, 2)
        tmp_105 = None
        tmp_107 = tmp_106.mean(dim=[-2, -1])
        return (tmp_106, tmp_107)