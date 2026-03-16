import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, in_1):
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
        tmp_19 = in_1
        tmp_20 = tmp_0[slice(None, None, None), slice(None, 80000, None)]
        tmp_0 = None
        tmp_21 = tmp_19.unsqueeze(1)
        tmp_19 = None
        tmp_22 = torch.conv1d(tmp_21, tmp_10, None, (5,), (0,), (1,), 1)
        tmp_21 = tmp_10 = None
        tmp_23 = torch.nn.functional.group_norm(tmp_22, 512, tmp_12, tmp_11, 1e-05)
        tmp_22 = tmp_12 = tmp_11 = None
        tmp_24 = torch.nn.functional.gelu(tmp_23)
        tmp_23 = None
        tmp_25 = torch.conv1d(tmp_24, tmp_13, None, (2,), (0,), (1,), 1)
        tmp_24 = tmp_13 = None
        tmp_26 = torch.nn.functional.gelu(tmp_25)
        tmp_25 = None
        tmp_27 = torch.conv1d(tmp_26, tmp_14, None, (2,), (0,), (1,), 1)
        tmp_26 = tmp_14 = None
        tmp_28 = torch.nn.functional.gelu(tmp_27)
        tmp_27 = None
        tmp_29 = torch.conv1d(tmp_28, tmp_15, None, (2,), (0,), (1,), 1)
        tmp_28 = tmp_15 = None
        tmp_30 = torch.nn.functional.gelu(tmp_29)
        tmp_29 = None
        tmp_31 = torch.conv1d(tmp_30, tmp_16, None, (2,), (0,), (1,), 1)
        tmp_30 = tmp_16 = None
        tmp_32 = torch.nn.functional.gelu(tmp_31)
        tmp_31 = None
        tmp_33 = torch.conv1d(tmp_32, tmp_17, None, (2,), (0,), (1,), 1)
        tmp_32 = tmp_17 = None
        tmp_34 = torch.nn.functional.gelu(tmp_33)
        tmp_33 = None
        tmp_35 = torch.conv1d(tmp_34, tmp_18, None, (2,), (0,), (1,), 1)
        tmp_34 = tmp_18 = None
        tmp_36 = torch.nn.functional.gelu(tmp_35)
        tmp_35 = None
        tmp_37 = tmp_36.transpose(1, 2)
        tmp_36 = None
        tmp_38 = torch.nn.functional.layer_norm(tmp_37, (512,), tmp_2, tmp_1, 1e-05)
        tmp_37 = tmp_2 = tmp_1 = None
        tmp_39 = torch.nn.functional.linear(tmp_38, tmp_4, tmp_3)
        tmp_38 = tmp_4 = tmp_3 = None
        tmp_40 = torch.nn.functional.dropout(tmp_39, 0.1, False, False)
        tmp_39 = None
        tmp_41 = tmp_40.transpose(-2, -1)
        tmp_42 = torch._weight_norm(tmp_8, tmp_7, 2)
        tmp_8 = tmp_7 = None
        tmp_43 = torch.conv1d(tmp_41, tmp_42, tmp_9, (1,), (64,), (1,), 16)
        tmp_41 = tmp_42 = tmp_9 = None
        tmp_44 = tmp_43[Ellipsis, slice(None, -1, None)]
        tmp_43 = None
        tmp_45 = torch.nn.functional.gelu(tmp_44)
        tmp_44 = None
        tmp_46 = tmp_45.transpose(-2, -1)
        tmp_45 = None
        tmp_47 = tmp_40 + tmp_46
        tmp_40 = tmp_46 = None
        tmp_48 = torch.nn.functional.layer_norm(tmp_47, (768,), tmp_6, tmp_5, 1e-05)
        tmp_47 = tmp_6 = tmp_5 = None
        tmp_49 = torch.nn.functional.dropout(tmp_48, 0.1, False, False)
        tmp_48 = None
        return (tmp_20, tmp_49)