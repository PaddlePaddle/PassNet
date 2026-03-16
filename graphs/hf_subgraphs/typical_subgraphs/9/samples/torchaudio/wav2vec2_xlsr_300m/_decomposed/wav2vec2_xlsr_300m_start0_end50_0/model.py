import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, w_26, w_27, w_28, w_29, w_30, w_31, w_32, w_33, w_34, w_35, w_36, in_0):
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
        tmp_37 = in_0
        tmp_38 = torch.nn.functional.layer_norm(tmp_37, (1, 80000))
        tmp_37 = None
        tmp_39 = tmp_38.unsqueeze(1)
        tmp_38 = None
        tmp_40 = torch.conv1d(tmp_39, tmp_10, tmp_9, (5,), (0,), (1,), 1)
        tmp_39 = tmp_10 = tmp_9 = None
        tmp_41 = tmp_40.transpose(-2, -1)
        tmp_40 = None
        tmp_42 = torch.nn.functional.layer_norm(tmp_41, (512,), tmp_12, tmp_11, 1e-05)
        tmp_41 = tmp_12 = tmp_11 = None
        tmp_43 = tmp_42.transpose(-2, -1)
        tmp_42 = None
        tmp_44 = torch.nn.functional.gelu(tmp_43)
        tmp_43 = None
        tmp_45 = torch.conv1d(tmp_44, tmp_14, tmp_13, (2,), (0,), (1,), 1)
        tmp_44 = tmp_14 = tmp_13 = None
        tmp_46 = tmp_45.transpose(-2, -1)
        tmp_45 = None
        tmp_47 = torch.nn.functional.layer_norm(tmp_46, (512,), tmp_16, tmp_15, 1e-05)
        tmp_46 = tmp_16 = tmp_15 = None
        tmp_48 = tmp_47.transpose(-2, -1)
        tmp_47 = None
        tmp_49 = torch.nn.functional.gelu(tmp_48)
        tmp_48 = None
        tmp_50 = torch.conv1d(tmp_49, tmp_18, tmp_17, (2,), (0,), (1,), 1)
        tmp_49 = tmp_18 = tmp_17 = None
        tmp_51 = tmp_50.transpose(-2, -1)
        tmp_50 = None
        tmp_52 = torch.nn.functional.layer_norm(tmp_51, (512,), tmp_20, tmp_19, 1e-05)
        tmp_51 = tmp_20 = tmp_19 = None
        tmp_53 = tmp_52.transpose(-2, -1)
        tmp_52 = None
        tmp_54 = torch.nn.functional.gelu(tmp_53)
        tmp_53 = None
        tmp_55 = torch.conv1d(tmp_54, tmp_22, tmp_21, (2,), (0,), (1,), 1)
        tmp_54 = tmp_22 = tmp_21 = None
        tmp_56 = tmp_55.transpose(-2, -1)
        tmp_55 = None
        tmp_57 = torch.nn.functional.layer_norm(tmp_56, (512,), tmp_24, tmp_23, 1e-05)
        tmp_56 = tmp_24 = tmp_23 = None
        tmp_58 = tmp_57.transpose(-2, -1)
        tmp_57 = None
        tmp_59 = torch.nn.functional.gelu(tmp_58)
        tmp_58 = None
        tmp_60 = torch.conv1d(tmp_59, tmp_26, tmp_25, (2,), (0,), (1,), 1)
        tmp_59 = tmp_26 = tmp_25 = None
        tmp_61 = tmp_60.transpose(-2, -1)
        tmp_60 = None
        tmp_62 = torch.nn.functional.layer_norm(tmp_61, (512,), tmp_28, tmp_27, 1e-05)
        tmp_61 = tmp_28 = tmp_27 = None
        tmp_63 = tmp_62.transpose(-2, -1)
        tmp_62 = None
        tmp_64 = torch.nn.functional.gelu(tmp_63)
        tmp_63 = None
        tmp_65 = torch.conv1d(tmp_64, tmp_30, tmp_29, (2,), (0,), (1,), 1)
        tmp_64 = tmp_30 = tmp_29 = None
        tmp_66 = tmp_65.transpose(-2, -1)
        tmp_65 = None
        tmp_67 = torch.nn.functional.layer_norm(tmp_66, (512,), tmp_32, tmp_31, 1e-05)
        tmp_66 = tmp_32 = tmp_31 = None
        tmp_68 = tmp_67.transpose(-2, -1)
        tmp_67 = None
        tmp_69 = torch.nn.functional.gelu(tmp_68)
        tmp_68 = None
        tmp_70 = torch.conv1d(tmp_69, tmp_34, tmp_33, (2,), (0,), (1,), 1)
        tmp_69 = tmp_34 = tmp_33 = None
        tmp_71 = tmp_70.transpose(-2, -1)
        tmp_70 = None
        tmp_72 = torch.nn.functional.layer_norm(tmp_71, (512,), tmp_36, tmp_35, 1e-05)
        tmp_71 = tmp_36 = tmp_35 = None
        tmp_73 = tmp_72.transpose(-2, -1)
        tmp_72 = None
        tmp_74 = torch.nn.functional.gelu(tmp_73)
        tmp_73 = None
        tmp_75 = tmp_74.transpose(1, 2)
        tmp_74 = None
        tmp_76 = torch.nn.functional.layer_norm(tmp_75, (512,), tmp_1, tmp_0, 1e-05)
        tmp_75 = tmp_1 = tmp_0 = None
        tmp_77 = torch.nn.functional.linear(tmp_76, tmp_3, tmp_2)
        tmp_76 = tmp_3 = tmp_2 = None
        tmp_78 = torch.nn.functional.dropout(tmp_77, 0.0, False, False)
        tmp_77 = None
        tmp_79 = tmp_78.transpose(-2, -1)
        tmp_80 = torch._weight_norm(tmp_7, tmp_6, 2)
        tmp_7 = tmp_6 = None
        tmp_81 = torch.conv1d(tmp_79, tmp_80, tmp_8, (1,), (64,), (1,), 16)
        tmp_79 = tmp_80 = tmp_8 = None
        tmp_82 = tmp_81[Ellipsis, slice(None, -1, None)]
        tmp_81 = None
        tmp_83 = torch.nn.functional.gelu(tmp_82)
        tmp_82 = None
        tmp_84 = tmp_83.transpose(-2, -1)
        tmp_83 = None
        tmp_85 = tmp_78 + tmp_84
        tmp_78 = tmp_84 = None
        tmp_86 = torch.nn.functional.dropout(tmp_85, 0.0, False, False)
        tmp_85 = None
        tmp_87 = torch.nn.functional.layer_norm(tmp_86, (1024,), tmp_5, tmp_4, 1e-05)
        tmp_5 = tmp_4 = None
        return (tmp_86, tmp_87)