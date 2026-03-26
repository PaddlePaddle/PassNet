import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41, in_42, in_43, in_44, in_45, in_46):
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
        tmp_42 = torch.nn.functional.linear(in_42, tmp_5, tmp_4)
        tmp_5 = tmp_4 = None
        tmp_43 = tmp_42.view(16, -1, 4, 32)
        tmp_42 = None
        tmp_44 = tmp_43.transpose(1, 2)
        tmp_43 = None
        tmp_45 = in_44.transpose(-1, -2)
        tmp_46 = torch.matmul(in_46, tmp_45)
        tmp_45 = None
        tmp_47 = tmp_46 / 5.656854249492381
        tmp_46 = None
        tmp_48 = tmp_47 + in_43
        tmp_47 = None
        tmp_49 = torch.nn.functional.softmax(tmp_48, dim=-1)
        tmp_48 = None
        tmp_50 = torch.nn.functional.dropout(tmp_49, 0.1, False, False)
        tmp_49 = None
        tmp_51 = torch.matmul(tmp_50, tmp_44)
        tmp_50 = tmp_44 = None
        tmp_52 = tmp_51.permute(0, 2, 1, 3)
        tmp_51 = None
        tmp_53 = tmp_52.contiguous()
        tmp_52 = None
        tmp_54 = tmp_53.view((16, 128, 128))
        tmp_53 = None
        tmp_55 = torch.nn.functional.linear(tmp_54, tmp_3, tmp_2)
        tmp_54 = tmp_3 = tmp_2 = None
        tmp_56 = tmp_55 + in_45
        tmp_55 = None
        tmp_57 = tmp_56 * tmp_1
        tmp_56 = tmp_1 = None
        tmp_58 = tmp_57 + tmp_0
        tmp_57 = tmp_0 = None
        tmp_59 = torch.nn.functional.linear(tmp_58, tmp_7, tmp_6)
        tmp_7 = tmp_6 = None
        tmp_60 = torch.nn.functional.gelu(tmp_59)
        tmp_59 = None
        tmp_61 = torch.nn.functional.linear(tmp_60, tmp_11, tmp_10)
        tmp_60 = tmp_11 = tmp_10 = None
        tmp_62 = tmp_61 + tmp_58
        tmp_61 = tmp_58 = None
        tmp_63 = tmp_62 * tmp_9
        tmp_62 = tmp_9 = None
        tmp_64 = tmp_63 + tmp_8
        tmp_63 = tmp_8 = None
        tmp_65 = torch.nn.functional.linear(tmp_64, tmp_13, tmp_12)
        tmp_13 = tmp_12 = None
        tmp_66 = torch.nn.functional.gelu(tmp_65)
        tmp_65 = None
        tmp_67 = torch.nn.functional.linear(tmp_66, tmp_17, tmp_16)
        tmp_66 = tmp_17 = tmp_16 = None
        tmp_68 = tmp_67 + tmp_64
        tmp_67 = tmp_64 = None
        tmp_69 = tmp_68 * tmp_15
        tmp_68 = tmp_15 = None
        tmp_70 = tmp_69 + tmp_14
        tmp_69 = tmp_14 = None
        tmp_71 = torch.nn.functional.linear(tmp_70, tmp_19, tmp_18)
        tmp_19 = tmp_18 = None
        tmp_72 = torch.nn.functional.gelu(tmp_71)
        tmp_71 = None
        tmp_73 = torch.nn.functional.linear(tmp_72, tmp_23, tmp_22)
        tmp_72 = tmp_23 = tmp_22 = None
        tmp_74 = tmp_73 + tmp_70
        tmp_73 = tmp_70 = None
        tmp_75 = tmp_74 * tmp_21
        tmp_74 = tmp_21 = None
        tmp_76 = tmp_75 + tmp_20
        tmp_75 = tmp_20 = None
        tmp_77 = torch.nn.functional.linear(tmp_76, tmp_25, tmp_24)
        tmp_25 = tmp_24 = None
        tmp_78 = torch.nn.functional.gelu(tmp_77)
        tmp_77 = None
        tmp_79 = torch.nn.functional.linear(tmp_78, tmp_33, tmp_32)
        tmp_78 = tmp_33 = tmp_32 = None
        tmp_80 = tmp_79 + tmp_76
        tmp_79 = tmp_76 = None
        tmp_81 = tmp_80 * tmp_27
        tmp_80 = tmp_27 = None
        tmp_82 = tmp_81 + tmp_26
        tmp_81 = tmp_26 = None
        tmp_83 = torch.nn.functional.linear(tmp_82, tmp_31, tmp_30)
        tmp_82 = tmp_31 = tmp_30 = None
        tmp_84 = torch.nn.functional.dropout(tmp_83, 0.1, False, False)
        tmp_83 = None
        tmp_85 = tmp_84 + in_42
        tmp_84 = None
        tmp_86 = tmp_85 * tmp_29
        tmp_85 = tmp_29 = None
        tmp_87 = tmp_86 + tmp_28
        tmp_86 = tmp_28 = None
        tmp_88 = torch.tensor(1000)
        tmp_88 = None
        tmp_89 = torch.nn.functional.linear(tmp_87, tmp_41, tmp_40)
        tmp_41 = tmp_40 = None
        tmp_90 = tmp_89 * tmp_39
        tmp_89 = tmp_39 = None
        tmp_91 = tmp_90 + tmp_38
        tmp_90 = tmp_38 = None
        tmp_92 = torch.nn.functional.linear(tmp_87, tmp_37, tmp_36)
        tmp_37 = tmp_36 = None
        tmp_93 = tmp_92 * tmp_35
        tmp_92 = tmp_35 = None
        tmp_94 = tmp_93 + tmp_34
        tmp_93 = tmp_34 = None
        return (tmp_91, tmp_94, tmp_87)