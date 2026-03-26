import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, w_26, w_27, w_28, w_29, in_0):
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
        tmp_30 = in_0
        tmp_31 = torch.nn.functional.layer_norm(tmp_30, (1, 80000))
        tmp_30 = None
        tmp_32 = tmp_31.unsqueeze(1)
        tmp_31 = None
        tmp_33 = torch.conv1d(tmp_32, tmp_9, None, (5,), (0,), (1,), 1)
        tmp_32 = tmp_9 = None
        tmp_34 = tmp_33.transpose(-2, -1)
        tmp_33 = None
        tmp_35 = torch.nn.functional.layer_norm(tmp_34, (512,), tmp_11, tmp_10, 1e-05)
        tmp_34 = tmp_11 = tmp_10 = None
        tmp_36 = tmp_35.transpose(-2, -1)
        tmp_35 = None
        tmp_37 = torch.nn.functional.gelu(tmp_36)
        tmp_36 = None
        tmp_38 = torch.conv1d(tmp_37, tmp_12, None, (2,), (0,), (1,), 1)
        tmp_37 = tmp_12 = None
        tmp_39 = tmp_38.transpose(-2, -1)
        tmp_38 = None
        tmp_40 = torch.nn.functional.layer_norm(tmp_39, (512,), tmp_14, tmp_13, 1e-05)
        tmp_39 = tmp_14 = tmp_13 = None
        tmp_41 = tmp_40.transpose(-2, -1)
        tmp_40 = None
        tmp_42 = torch.nn.functional.gelu(tmp_41)
        tmp_41 = None
        tmp_43 = torch.conv1d(tmp_42, tmp_15, None, (2,), (0,), (1,), 1)
        tmp_42 = tmp_15 = None
        tmp_44 = tmp_43.transpose(-2, -1)
        tmp_43 = None
        tmp_45 = torch.nn.functional.layer_norm(tmp_44, (512,), tmp_17, tmp_16, 1e-05)
        tmp_44 = tmp_17 = tmp_16 = None
        tmp_46 = tmp_45.transpose(-2, -1)
        tmp_45 = None
        tmp_47 = torch.nn.functional.gelu(tmp_46)
        tmp_46 = None
        tmp_48 = torch.conv1d(tmp_47, tmp_18, None, (2,), (0,), (1,), 1)
        tmp_47 = tmp_18 = None
        tmp_49 = tmp_48.transpose(-2, -1)
        tmp_48 = None
        tmp_50 = torch.nn.functional.layer_norm(tmp_49, (512,), tmp_20, tmp_19, 1e-05)
        tmp_49 = tmp_20 = tmp_19 = None
        tmp_51 = tmp_50.transpose(-2, -1)
        tmp_50 = None
        tmp_52 = torch.nn.functional.gelu(tmp_51)
        tmp_51 = None
        tmp_53 = torch.conv1d(tmp_52, tmp_21, None, (2,), (0,), (1,), 1)
        tmp_52 = tmp_21 = None
        tmp_54 = tmp_53.transpose(-2, -1)
        tmp_53 = None
        tmp_55 = torch.nn.functional.layer_norm(tmp_54, (512,), tmp_23, tmp_22, 1e-05)
        tmp_54 = tmp_23 = tmp_22 = None
        tmp_56 = tmp_55.transpose(-2, -1)
        tmp_55 = None
        tmp_57 = torch.nn.functional.gelu(tmp_56)
        tmp_56 = None
        tmp_58 = torch.conv1d(tmp_57, tmp_24, None, (2,), (0,), (1,), 1)
        tmp_57 = tmp_24 = None
        tmp_59 = tmp_58.transpose(-2, -1)
        tmp_58 = None
        tmp_60 = torch.nn.functional.layer_norm(tmp_59, (512,), tmp_26, tmp_25, 1e-05)
        tmp_59 = tmp_26 = tmp_25 = None
        tmp_61 = tmp_60.transpose(-2, -1)
        tmp_60 = None
        tmp_62 = torch.nn.functional.gelu(tmp_61)
        tmp_61 = None
        tmp_63 = torch.conv1d(tmp_62, tmp_27, None, (2,), (0,), (1,), 1)
        tmp_62 = tmp_27 = None
        tmp_64 = tmp_63.transpose(-2, -1)
        tmp_63 = None
        tmp_65 = torch.nn.functional.layer_norm(tmp_64, (512,), tmp_29, tmp_28, 1e-05)
        tmp_64 = tmp_29 = tmp_28 = None
        tmp_66 = tmp_65.transpose(-2, -1)
        tmp_65 = None
        tmp_67 = torch.nn.functional.gelu(tmp_66)
        tmp_66 = None
        tmp_68 = tmp_67.transpose(1, 2)
        tmp_67 = None
        tmp_69 = torch.nn.functional.layer_norm(tmp_68, (512,), tmp_1, tmp_0, 1e-05)
        tmp_68 = tmp_1 = tmp_0 = None
        tmp_70 = torch.nn.functional.linear(tmp_69, tmp_3, tmp_2)
        tmp_69 = tmp_3 = tmp_2 = None
        tmp_71 = torch.nn.functional.dropout(tmp_70, 0.0, False, False)
        tmp_70 = None
        tmp_72 = tmp_71.transpose(-2, -1)
        tmp_73 = torch._weight_norm(tmp_7, tmp_6, 2)
        tmp_7 = tmp_6 = None
        tmp_74 = torch.conv1d(tmp_72, tmp_73, tmp_8, (1,), (64,), (1,), 16)
        tmp_72 = tmp_73 = tmp_8 = None
        tmp_75 = tmp_74[Ellipsis, slice(None, -1, None)]
        tmp_74 = None
        tmp_76 = torch.nn.functional.gelu(tmp_75)
        tmp_75 = None
        tmp_77 = tmp_76.transpose(-2, -1)
        tmp_76 = None
        tmp_78 = tmp_71 + tmp_77
        tmp_71 = tmp_77 = None
        tmp_79 = torch.nn.functional.dropout(tmp_78, 0.0, False, False)
        tmp_78 = None
        tmp_80 = torch.nn.functional.layer_norm(tmp_79, (1024,), tmp_5, tmp_4, 1e-05)
        tmp_5 = tmp_4 = None
        return (tmp_79, tmp_80)