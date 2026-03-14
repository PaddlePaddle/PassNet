import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, in_0):
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
        tmp_18 = in_0
        tmp_19 = tmp_18.unsqueeze(1)
        tmp_18 = None
        tmp_20 = torch.conv1d(tmp_19, tmp_9, None, (5,), (0,), (1,), 1)
        tmp_19 = tmp_9 = None
        tmp_21 = torch.nn.functional.group_norm(tmp_20, 512, tmp_11, tmp_10, 1e-05)
        tmp_20 = tmp_11 = tmp_10 = None
        tmp_22 = torch.nn.functional.gelu(tmp_21)
        tmp_21 = None
        tmp_23 = torch.conv1d(tmp_22, tmp_12, None, (2,), (0,), (1,), 1)
        tmp_22 = tmp_12 = None
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
        tmp_35 = tmp_34.transpose(1, 2)
        tmp_34 = None
        tmp_36 = torch.nn.functional.layer_norm(tmp_35, (512,), tmp_1, tmp_0, 1e-05)
        tmp_35 = tmp_1 = tmp_0 = None
        tmp_37 = torch.nn.functional.linear(tmp_36, tmp_3, tmp_2)
        tmp_36 = tmp_3 = tmp_2 = None
        tmp_38 = torch.nn.functional.dropout(tmp_37, 0.1, False, False)
        tmp_37 = None
        tmp_39 = tmp_38.transpose(-2, -1)
        tmp_40 = torch._weight_norm(tmp_7, tmp_6, 2)
        tmp_7 = tmp_6 = None
        tmp_41 = torch.conv1d(tmp_39, tmp_40, tmp_8, (1,), (64,), (1,), 16)
        tmp_39 = tmp_40 = tmp_8 = None
        tmp_42 = tmp_41[Ellipsis, slice(None, -1, None)]
        tmp_41 = None
        tmp_43 = torch.nn.functional.gelu(tmp_42)
        tmp_42 = None
        tmp_44 = tmp_43.transpose(-2, -1)
        tmp_43 = None
        tmp_45 = tmp_38 + tmp_44
        tmp_38 = tmp_44 = None
        tmp_46 = torch.nn.functional.layer_norm(tmp_45, (1024,), tmp_5, tmp_4, 1e-05)
        tmp_45 = tmp_5 = tmp_4 = None
        tmp_47 = torch.nn.functional.dropout(tmp_46, 0.0, False, False)
        tmp_46 = None
        return (tmp_47,)