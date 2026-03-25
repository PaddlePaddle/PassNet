import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, w_26, w_27, w_28, w_29):
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
        tmp_31 = tmp_0[slice(None, None, None), None]
        tmp_0 = None
        tmp_32 = torch.conv1d(tmp_31, tmp_6, None, (5,), (0,), (1,), 1)
        tmp_31 = tmp_6 = None
        tmp_33 = tmp_32.transpose(-2, -1)
        tmp_32 = None
        tmp_34 = torch.nn.functional.layer_norm(tmp_33, (512,), tmp_8, tmp_7, 1e-05)
        tmp_33 = tmp_8 = tmp_7 = None
        tmp_35 = tmp_34.transpose(-2, -1)
        tmp_34 = None
        tmp_36 = torch.nn.functional.gelu(tmp_35)
        tmp_35 = None
        tmp_37 = torch.conv1d(tmp_36, tmp_9, None, (2,), (0,), (1,), 1)
        tmp_36 = tmp_9 = None
        tmp_38 = tmp_37.transpose(-2, -1)
        tmp_37 = None
        tmp_39 = torch.nn.functional.layer_norm(tmp_38, (512,), tmp_11, tmp_10, 1e-05)
        tmp_38 = tmp_11 = tmp_10 = None
        tmp_40 = tmp_39.transpose(-2, -1)
        tmp_39 = None
        tmp_41 = torch.nn.functional.gelu(tmp_40)
        tmp_40 = None
        tmp_42 = torch.conv1d(tmp_41, tmp_12, None, (2,), (0,), (1,), 1)
        tmp_41 = tmp_12 = None
        tmp_43 = tmp_42.transpose(-2, -1)
        tmp_42 = None
        tmp_44 = torch.nn.functional.layer_norm(tmp_43, (512,), tmp_14, tmp_13, 1e-05)
        tmp_43 = tmp_14 = tmp_13 = None
        tmp_45 = tmp_44.transpose(-2, -1)
        tmp_44 = None
        tmp_46 = torch.nn.functional.gelu(tmp_45)
        tmp_45 = None
        tmp_47 = torch.conv1d(tmp_46, tmp_15, None, (2,), (0,), (1,), 1)
        tmp_46 = tmp_15 = None
        tmp_48 = tmp_47.transpose(-2, -1)
        tmp_47 = None
        tmp_49 = torch.nn.functional.layer_norm(tmp_48, (512,), tmp_17, tmp_16, 1e-05)
        tmp_48 = tmp_17 = tmp_16 = None
        tmp_50 = tmp_49.transpose(-2, -1)
        tmp_49 = None
        tmp_51 = torch.nn.functional.gelu(tmp_50)
        tmp_50 = None
        tmp_52 = torch.conv1d(tmp_51, tmp_18, None, (2,), (0,), (1,), 1)
        tmp_51 = tmp_18 = None
        tmp_53 = tmp_52.transpose(-2, -1)
        tmp_52 = None
        tmp_54 = torch.nn.functional.layer_norm(tmp_53, (512,), tmp_20, tmp_19, 1e-05)
        tmp_53 = tmp_20 = tmp_19 = None
        tmp_55 = tmp_54.transpose(-2, -1)
        tmp_54 = None
        tmp_56 = torch.nn.functional.gelu(tmp_55)
        tmp_55 = None
        tmp_57 = torch.conv1d(tmp_56, tmp_21, None, (2,), (0,), (1,), 1)
        tmp_56 = tmp_21 = None
        tmp_58 = tmp_57.transpose(-2, -1)
        tmp_57 = None
        tmp_59 = torch.nn.functional.layer_norm(tmp_58, (512,), tmp_23, tmp_22, 1e-05)
        tmp_58 = tmp_23 = tmp_22 = None
        tmp_60 = tmp_59.transpose(-2, -1)
        tmp_59 = None
        tmp_61 = torch.nn.functional.gelu(tmp_60)
        tmp_60 = None
        tmp_62 = torch.conv1d(tmp_61, tmp_24, None, (2,), (0,), (1,), 1)
        tmp_61 = tmp_24 = None
        tmp_63 = tmp_62.transpose(-2, -1)
        tmp_62 = None
        tmp_64 = torch.nn.functional.layer_norm(tmp_63, (512,), tmp_26, tmp_25, 1e-05)
        tmp_63 = tmp_26 = tmp_25 = None
        tmp_65 = tmp_64.transpose(-2, -1)
        tmp_64 = None
        tmp_66 = torch.nn.functional.gelu(tmp_65)
        tmp_65 = None
        tmp_67 = tmp_66.transpose(1, 2)
        tmp_66 = None
        tmp_68 = torch.nn.functional.layer_norm(tmp_67, (512,), tmp_28, tmp_27, 1e-05)
        tmp_67 = tmp_28 = tmp_27 = None
        tmp_69 = torch.nn.functional.linear(tmp_68, tmp_30, tmp_29)
        tmp_68 = tmp_30 = tmp_29 = None
        tmp_70 = torch.nn.functional.dropout(tmp_69, 0.05, False, False)
        tmp_69 = None
        tmp_71 = tmp_70.transpose(1, 2)
        tmp_72 = torch._weight_norm(tmp_4, tmp_3, 2)
        tmp_4 = tmp_3 = None
        tmp_73 = torch.conv1d(tmp_71, tmp_72, tmp_5, (1,), (64,), (1,), 16)
        tmp_71 = tmp_72 = tmp_5 = None
        tmp_74 = tmp_73[slice(None, None, None), slice(None, None, None), slice(None, -1, None)]
        tmp_73 = None
        tmp_75 = torch.nn.functional.gelu(tmp_74)
        tmp_74 = None
        tmp_76 = tmp_75.transpose(1, 2)
        tmp_75 = None
        tmp_77 = tmp_70 + tmp_76
        tmp_70 = tmp_76 = None
        tmp_78 = torch.nn.functional.dropout(tmp_77, 0.05, False, False)
        tmp_77 = None
        tmp_79 = torch.rand([])
        tmp_79 = None
        tmp_80 = torch.nn.functional.layer_norm(tmp_78, (1024,), tmp_2, tmp_1, 1e-05)
        tmp_2 = tmp_1 = None
        return (tmp_78, tmp_80)