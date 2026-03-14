import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, w_26, w_27, w_28, w_29, w_30, w_31, w_32, w_33, w_34, w_35, w_36):
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
        tmp_38 = tmp_0[slice(None, None, None), None]
        tmp_0 = None
        tmp_39 = torch.conv1d(tmp_38, tmp_7, tmp_6, (5,), (0,), (1,), 1)
        tmp_38 = tmp_7 = tmp_6 = None
        tmp_40 = tmp_39.transpose(-2, -1)
        tmp_39 = None
        tmp_41 = torch.nn.functional.layer_norm(tmp_40, (512,), tmp_9, tmp_8, 1e-05)
        tmp_40 = tmp_9 = tmp_8 = None
        tmp_42 = tmp_41.transpose(-2, -1)
        tmp_41 = None
        tmp_43 = torch.nn.functional.gelu(tmp_42)
        tmp_42 = None
        tmp_44 = torch.conv1d(tmp_43, tmp_11, tmp_10, (2,), (0,), (1,), 1)
        tmp_43 = tmp_11 = tmp_10 = None
        tmp_45 = tmp_44.transpose(-2, -1)
        tmp_44 = None
        tmp_46 = torch.nn.functional.layer_norm(tmp_45, (512,), tmp_13, tmp_12, 1e-05)
        tmp_45 = tmp_13 = tmp_12 = None
        tmp_47 = tmp_46.transpose(-2, -1)
        tmp_46 = None
        tmp_48 = torch.nn.functional.gelu(tmp_47)
        tmp_47 = None
        tmp_49 = torch.conv1d(tmp_48, tmp_15, tmp_14, (2,), (0,), (1,), 1)
        tmp_48 = tmp_15 = tmp_14 = None
        tmp_50 = tmp_49.transpose(-2, -1)
        tmp_49 = None
        tmp_51 = torch.nn.functional.layer_norm(tmp_50, (512,), tmp_17, tmp_16, 1e-05)
        tmp_50 = tmp_17 = tmp_16 = None
        tmp_52 = tmp_51.transpose(-2, -1)
        tmp_51 = None
        tmp_53 = torch.nn.functional.gelu(tmp_52)
        tmp_52 = None
        tmp_54 = torch.conv1d(tmp_53, tmp_19, tmp_18, (2,), (0,), (1,), 1)
        tmp_53 = tmp_19 = tmp_18 = None
        tmp_55 = tmp_54.transpose(-2, -1)
        tmp_54 = None
        tmp_56 = torch.nn.functional.layer_norm(tmp_55, (512,), tmp_21, tmp_20, 1e-05)
        tmp_55 = tmp_21 = tmp_20 = None
        tmp_57 = tmp_56.transpose(-2, -1)
        tmp_56 = None
        tmp_58 = torch.nn.functional.gelu(tmp_57)
        tmp_57 = None
        tmp_59 = torch.conv1d(tmp_58, tmp_23, tmp_22, (2,), (0,), (1,), 1)
        tmp_58 = tmp_23 = tmp_22 = None
        tmp_60 = tmp_59.transpose(-2, -1)
        tmp_59 = None
        tmp_61 = torch.nn.functional.layer_norm(tmp_60, (512,), tmp_25, tmp_24, 1e-05)
        tmp_60 = tmp_25 = tmp_24 = None
        tmp_62 = tmp_61.transpose(-2, -1)
        tmp_61 = None
        tmp_63 = torch.nn.functional.gelu(tmp_62)
        tmp_62 = None
        tmp_64 = torch.conv1d(tmp_63, tmp_27, tmp_26, (2,), (0,), (1,), 1)
        tmp_63 = tmp_27 = tmp_26 = None
        tmp_65 = tmp_64.transpose(-2, -1)
        tmp_64 = None
        tmp_66 = torch.nn.functional.layer_norm(tmp_65, (512,), tmp_29, tmp_28, 1e-05)
        tmp_65 = tmp_29 = tmp_28 = None
        tmp_67 = tmp_66.transpose(-2, -1)
        tmp_66 = None
        tmp_68 = torch.nn.functional.gelu(tmp_67)
        tmp_67 = None
        tmp_69 = torch.conv1d(tmp_68, tmp_31, tmp_30, (2,), (0,), (1,), 1)
        tmp_68 = tmp_31 = tmp_30 = None
        tmp_70 = tmp_69.transpose(-2, -1)
        tmp_69 = None
        tmp_71 = torch.nn.functional.layer_norm(tmp_70, (512,), tmp_33, tmp_32, 1e-05)
        tmp_70 = tmp_33 = tmp_32 = None
        tmp_72 = tmp_71.transpose(-2, -1)
        tmp_71 = None
        tmp_73 = torch.nn.functional.gelu(tmp_72)
        tmp_72 = None
        tmp_74 = tmp_73.transpose(1, 2)
        tmp_73 = None
        tmp_75 = torch.nn.functional.layer_norm(tmp_74, (512,), tmp_35, tmp_34, 1e-05)
        tmp_74 = tmp_35 = tmp_34 = None
        tmp_76 = torch.nn.functional.linear(tmp_75, tmp_37, tmp_36)
        tmp_75 = tmp_37 = tmp_36 = None
        tmp_77 = torch.nn.functional.dropout(tmp_76, 0.05, False, False)
        tmp_76 = None
        tmp_78 = tmp_77.transpose(1, 2)
        tmp_79 = torch._weight_norm(tmp_4, tmp_3, 2)
        tmp_4 = tmp_3 = None
        tmp_80 = torch.conv1d(tmp_78, tmp_79, tmp_5, (1,), (64,), (1,), 16)
        tmp_78 = tmp_79 = tmp_5 = None
        tmp_81 = tmp_80[slice(None, None, None), slice(None, None, None), slice(None, -1, None)]
        tmp_80 = None
        tmp_82 = torch.nn.functional.gelu(tmp_81)
        tmp_81 = None
        tmp_83 = tmp_82.transpose(1, 2)
        tmp_82 = None
        tmp_84 = tmp_77 + tmp_83
        tmp_77 = tmp_83 = None
        tmp_85 = torch.nn.functional.dropout(tmp_84, 0.05, False, False)
        tmp_84 = None
        tmp_86 = torch.rand([])
        tmp_86 = None
        tmp_87 = torch.nn.functional.layer_norm(tmp_85, (1024,), tmp_2, tmp_1, 1e-05)
        tmp_2 = tmp_1 = None
        return (tmp_85, tmp_87)