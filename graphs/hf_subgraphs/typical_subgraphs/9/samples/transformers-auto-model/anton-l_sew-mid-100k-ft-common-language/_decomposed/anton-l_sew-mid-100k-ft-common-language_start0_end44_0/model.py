import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23):
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
        tmp_25 = tmp_0[slice(None, None, None), None]
        tmp_0 = None
        tmp_26 = torch.conv1d(tmp_25, tmp_6, None, (5,), (0,), (1,), 1)
        tmp_25 = tmp_6 = None
        tmp_27 = torch.nn.functional.group_norm(tmp_26, 64, tmp_8, tmp_7, 1e-05)
        tmp_26 = tmp_8 = tmp_7 = None
        tmp_28 = torch.nn.functional.gelu(tmp_27)
        tmp_27 = None
        tmp_29 = torch.conv1d(tmp_28, tmp_12, None, (2,), (0,), (1,), 1)
        tmp_28 = tmp_12 = None
        tmp_30 = torch.nn.functional.gelu(tmp_29)
        tmp_29 = None
        tmp_31 = torch.conv1d(tmp_30, tmp_13, None, (1,), (0,), (1,), 1)
        tmp_30 = tmp_13 = None
        tmp_32 = torch.nn.functional.gelu(tmp_31)
        tmp_31 = None
        tmp_33 = torch.conv1d(tmp_32, tmp_14, None, (2,), (0,), (1,), 1)
        tmp_32 = tmp_14 = None
        tmp_34 = torch.nn.functional.gelu(tmp_33)
        tmp_33 = None
        tmp_35 = torch.conv1d(tmp_34, tmp_15, None, (1,), (0,), (1,), 1)
        tmp_34 = tmp_15 = None
        tmp_36 = torch.nn.functional.gelu(tmp_35)
        tmp_35 = None
        tmp_37 = torch.conv1d(tmp_36, tmp_16, None, (2,), (0,), (1,), 1)
        tmp_36 = tmp_16 = None
        tmp_38 = torch.nn.functional.gelu(tmp_37)
        tmp_37 = None
        tmp_39 = torch.conv1d(tmp_38, tmp_17, None, (1,), (0,), (1,), 1)
        tmp_38 = tmp_17 = None
        tmp_40 = torch.nn.functional.gelu(tmp_39)
        tmp_39 = None
        tmp_41 = torch.conv1d(tmp_40, tmp_18, None, (2,), (0,), (1,), 1)
        tmp_40 = tmp_18 = None
        tmp_42 = torch.nn.functional.gelu(tmp_41)
        tmp_41 = None
        tmp_43 = torch.conv1d(tmp_42, tmp_19, None, (1,), (0,), (1,), 1)
        tmp_42 = tmp_19 = None
        tmp_44 = torch.nn.functional.gelu(tmp_43)
        tmp_43 = None
        tmp_45 = torch.conv1d(tmp_44, tmp_20, None, (2,), (0,), (1,), 1)
        tmp_44 = tmp_20 = None
        tmp_46 = torch.nn.functional.gelu(tmp_45)
        tmp_45 = None
        tmp_47 = torch.conv1d(tmp_46, tmp_9, None, (1,), (0,), (1,), 1)
        tmp_46 = tmp_9 = None
        tmp_48 = torch.nn.functional.gelu(tmp_47)
        tmp_47 = None
        tmp_49 = torch.conv1d(tmp_48, tmp_10, None, (2,), (0,), (1,), 1)
        tmp_48 = tmp_10 = None
        tmp_50 = torch.nn.functional.gelu(tmp_49)
        tmp_49 = None
        tmp_51 = torch.conv1d(tmp_50, tmp_11, None, (1,), (0,), (1,), 1)
        tmp_50 = tmp_11 = None
        tmp_52 = torch.nn.functional.gelu(tmp_51)
        tmp_51 = None
        tmp_53 = tmp_52.transpose(1, 2)
        tmp_52 = None
        tmp_54 = torch.nn.functional.layer_norm(tmp_53, (512,), tmp_24, tmp_23, 1e-05)
        tmp_53 = tmp_24 = tmp_23 = None
        tmp_55 = torch.nn.functional.linear(tmp_54, tmp_22, tmp_21)
        tmp_54 = tmp_22 = tmp_21 = None
        tmp_56 = torch.nn.functional.dropout(tmp_55, 0.1, False, False)
        tmp_55 = None
        tmp_57 = tmp_56.transpose(1, 2)
        tmp_56 = None
        tmp_58 = torch._weight_norm(tmp_4, tmp_3, 2)
        tmp_4 = tmp_3 = None
        tmp_59 = torch.conv1d(tmp_57, tmp_58, tmp_5, (2,), (15,), (1,), 16)
        tmp_58 = tmp_5 = None
        tmp_60 = torch.nn.functional.gelu(tmp_59)
        tmp_59 = None
        tmp_61 = torch.avg_pool1d(tmp_57, (2,), (2,), (0,), False, True)
        tmp_57 = None
        tmp_62 = tmp_61[Ellipsis, slice(None, 124, None)]
        tmp_61 = None
        tmp_63 = tmp_60[Ellipsis, slice(None, 124, None)]
        tmp_60 = None
        tmp_64 = tmp_62 + tmp_63
        tmp_62 = tmp_63 = None
        tmp_65 = tmp_64.transpose(1, 2)
        tmp_64 = None
        tmp_66 = torch.nn.functional.layer_norm(tmp_65, (768,), tmp_2, tmp_1, 1e-05)
        tmp_65 = tmp_2 = tmp_1 = None
        tmp_67 = torch.nn.functional.dropout(tmp_66, 0.1, False, False)
        tmp_66 = None
        tmp_68 = torch.rand([])
        tmp_68 = None
        return (tmp_67,)