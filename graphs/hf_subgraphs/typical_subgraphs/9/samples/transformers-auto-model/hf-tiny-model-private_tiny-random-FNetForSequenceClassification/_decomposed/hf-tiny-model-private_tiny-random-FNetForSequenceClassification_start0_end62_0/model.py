import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, w_26, w_27, w_28, w_29, w_30, w_31, w_32, w_33, w_34, w_35, w_36, w_37, w_38, w_39, w_40, w_41, w_42, w_43, w_44, w_45, w_46, w_47, w_48, w_49, in_1):
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
        tmp_51 = in_1
        tmp_52 = tmp_1[slice(None, None, None), slice(None, 13, None)]
        tmp_1 = None
        tmp_53 = torch.nn.functional.embedding(tmp_0, tmp_8, 3, None, 2.0, False, False)
        tmp_0 = tmp_8 = None
        tmp_54 = torch.nn.functional.embedding(tmp_51, tmp_7, None, None, 2.0, False, False)
        tmp_51 = tmp_7 = None
        tmp_55 = tmp_53 + tmp_54
        tmp_53 = tmp_54 = None
        tmp_56 = torch.nn.functional.embedding(tmp_52, tmp_4, None, None, 2.0, False, False)
        tmp_52 = tmp_4 = None
        tmp_55 += tmp_56
        tmp_57 = tmp_55
        tmp_55 = tmp_56 = None
        tmp_58 = torch.nn.functional.layer_norm(tmp_57, (32,), tmp_3, tmp_2, 1e-12)
        tmp_57 = tmp_3 = tmp_2 = None
        tmp_59 = torch.nn.functional.linear(tmp_58, tmp_6, tmp_5)
        tmp_58 = tmp_6 = tmp_5 = None
        tmp_60 = torch.nn.functional.dropout(tmp_59, 0.1, False, False)
        tmp_59 = None
        tmp_61 = torch.fft.fftn(tmp_60, dim=(1, 2))
        tmp_62 = tmp_61.real
        tmp_61 = None
        tmp_63 = tmp_60 + tmp_62
        tmp_60 = tmp_62 = None
        tmp_64 = torch.nn.functional.layer_norm(tmp_63, (32,), tmp_10, tmp_9, 1e-12)
        tmp_63 = tmp_10 = tmp_9 = None
        tmp_65 = torch.nn.functional.linear(tmp_64, tmp_12, tmp_11)
        tmp_12 = tmp_11 = None
        tmp_66 = torch.nn.functional.gelu(tmp_65)
        tmp_65 = None
        tmp_67 = torch.nn.functional.linear(tmp_66, tmp_16, tmp_15)
        tmp_66 = tmp_16 = tmp_15 = None
        tmp_68 = torch.nn.functional.dropout(tmp_67, 0.1, False, False)
        tmp_67 = None
        tmp_69 = tmp_68 + tmp_64
        tmp_68 = tmp_64 = None
        tmp_70 = torch.nn.functional.layer_norm(tmp_69, (32,), tmp_14, tmp_13, 1e-12)
        tmp_69 = tmp_14 = tmp_13 = None
        tmp_71 = torch.fft.fftn(tmp_70, dim=(1, 2))
        tmp_72 = tmp_71.real
        tmp_71 = None
        tmp_73 = tmp_70 + tmp_72
        tmp_70 = tmp_72 = None
        tmp_74 = torch.nn.functional.layer_norm(tmp_73, (32,), tmp_18, tmp_17, 1e-12)
        tmp_73 = tmp_18 = tmp_17 = None
        tmp_75 = torch.nn.functional.linear(tmp_74, tmp_20, tmp_19)
        tmp_20 = tmp_19 = None
        tmp_76 = torch.nn.functional.gelu(tmp_75)
        tmp_75 = None
        tmp_77 = torch.nn.functional.linear(tmp_76, tmp_24, tmp_23)
        tmp_76 = tmp_24 = tmp_23 = None
        tmp_78 = torch.nn.functional.dropout(tmp_77, 0.1, False, False)
        tmp_77 = None
        tmp_79 = tmp_78 + tmp_74
        tmp_78 = tmp_74 = None
        tmp_80 = torch.nn.functional.layer_norm(tmp_79, (32,), tmp_22, tmp_21, 1e-12)
        tmp_79 = tmp_22 = tmp_21 = None
        tmp_81 = torch.fft.fftn(tmp_80, dim=(1, 2))
        tmp_82 = tmp_81.real
        tmp_81 = None
        tmp_83 = tmp_80 + tmp_82
        tmp_80 = tmp_82 = None
        tmp_84 = torch.nn.functional.layer_norm(tmp_83, (32,), tmp_26, tmp_25, 1e-12)
        tmp_83 = tmp_26 = tmp_25 = None
        tmp_85 = torch.nn.functional.linear(tmp_84, tmp_28, tmp_27)
        tmp_28 = tmp_27 = None
        tmp_86 = torch.nn.functional.gelu(tmp_85)
        tmp_85 = None
        tmp_87 = torch.nn.functional.linear(tmp_86, tmp_32, tmp_31)
        tmp_86 = tmp_32 = tmp_31 = None
        tmp_88 = torch.nn.functional.dropout(tmp_87, 0.1, False, False)
        tmp_87 = None
        tmp_89 = tmp_88 + tmp_84
        tmp_88 = tmp_84 = None
        tmp_90 = torch.nn.functional.layer_norm(tmp_89, (32,), tmp_30, tmp_29, 1e-12)
        tmp_89 = tmp_30 = tmp_29 = None
        tmp_91 = torch.fft.fftn(tmp_90, dim=(1, 2))
        tmp_92 = tmp_91.real
        tmp_91 = None
        tmp_93 = tmp_90 + tmp_92
        tmp_90 = tmp_92 = None
        tmp_94 = torch.nn.functional.layer_norm(tmp_93, (32,), tmp_34, tmp_33, 1e-12)
        tmp_93 = tmp_34 = tmp_33 = None
        tmp_95 = torch.nn.functional.linear(tmp_94, tmp_36, tmp_35)
        tmp_36 = tmp_35 = None
        tmp_96 = torch.nn.functional.gelu(tmp_95)
        tmp_95 = None
        tmp_97 = torch.nn.functional.linear(tmp_96, tmp_40, tmp_39)
        tmp_96 = tmp_40 = tmp_39 = None
        tmp_98 = torch.nn.functional.dropout(tmp_97, 0.1, False, False)
        tmp_97 = None
        tmp_99 = tmp_98 + tmp_94
        tmp_98 = tmp_94 = None
        tmp_100 = torch.nn.functional.layer_norm(tmp_99, (32,), tmp_38, tmp_37, 1e-12)
        tmp_99 = tmp_38 = tmp_37 = None
        tmp_101 = torch.fft.fftn(tmp_100, dim=(1, 2))
        tmp_102 = tmp_101.real
        tmp_101 = None
        tmp_103 = tmp_100 + tmp_102
        tmp_100 = tmp_102 = None
        tmp_104 = torch.nn.functional.layer_norm(tmp_103, (32,), tmp_42, tmp_41, 1e-12)
        tmp_103 = tmp_42 = tmp_41 = None
        tmp_105 = torch.nn.functional.linear(tmp_104, tmp_44, tmp_43)
        tmp_44 = tmp_43 = None
        tmp_106 = torch.nn.functional.gelu(tmp_105)
        tmp_105 = None
        tmp_107 = torch.nn.functional.linear(tmp_106, tmp_48, tmp_47)
        tmp_106 = tmp_48 = tmp_47 = None
        tmp_108 = torch.nn.functional.dropout(tmp_107, 0.1, False, False)
        tmp_107 = None
        tmp_109 = tmp_108 + tmp_104
        tmp_108 = tmp_104 = None
        tmp_110 = torch.nn.functional.layer_norm(tmp_109, (32,), tmp_46, tmp_45, 1e-12)
        tmp_109 = tmp_46 = tmp_45 = None
        tmp_111 = tmp_110[slice(None, None, None), 0]
        tmp_112 = torch.nn.functional.linear(tmp_111, tmp_50, tmp_49)
        tmp_111 = tmp_50 = tmp_49 = None
        tmp_113 = torch.tanh(tmp_112)
        tmp_112 = None
        return (tmp_110, tmp_113)