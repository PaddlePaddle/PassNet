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
        tmp_18 = in_0.unsqueeze(1)
        tmp_19 = torch.conv1d(tmp_18, tmp_9, None, (5,), (0,), (1,), 1)
        tmp_18 = tmp_9 = None
        tmp_20 = torch.nn.functional.group_norm(tmp_19, 512, tmp_11, tmp_10, 1e-05)
        tmp_19 = tmp_11 = tmp_10 = None
        tmp_21 = torch.nn.functional.gelu(tmp_20)
        tmp_20 = None
        tmp_22 = torch.conv1d(tmp_21, tmp_12, None, (2,), (0,), (1,), 1)
        tmp_21 = tmp_12 = None
        tmp_23 = torch.nn.functional.gelu(tmp_22)
        tmp_22 = None
        tmp_24 = torch.conv1d(tmp_23, tmp_13, None, (2,), (0,), (1,), 1)
        tmp_23 = tmp_13 = None
        tmp_25 = torch.nn.functional.gelu(tmp_24)
        tmp_24 = None
        tmp_26 = torch.conv1d(tmp_25, tmp_14, None, (2,), (0,), (1,), 1)
        tmp_25 = tmp_14 = None
        tmp_27 = torch.nn.functional.gelu(tmp_26)
        tmp_26 = None
        tmp_28 = torch.conv1d(tmp_27, tmp_15, None, (2,), (0,), (1,), 1)
        tmp_27 = tmp_15 = None
        tmp_29 = torch.nn.functional.gelu(tmp_28)
        tmp_28 = None
        tmp_30 = torch.conv1d(tmp_29, tmp_16, None, (2,), (0,), (1,), 1)
        tmp_29 = tmp_16 = None
        tmp_31 = torch.nn.functional.gelu(tmp_30)
        tmp_30 = None
        tmp_32 = torch.conv1d(tmp_31, tmp_17, None, (2,), (0,), (1,), 1)
        tmp_31 = tmp_17 = None
        tmp_33 = torch.nn.functional.gelu(tmp_32)
        tmp_32 = None
        tmp_34 = tmp_33.transpose(1, 2)
        tmp_33 = None
        tmp_35 = torch.nn.functional.layer_norm(tmp_34, (512,), tmp_1, tmp_0, 1e-05)
        tmp_34 = tmp_1 = tmp_0 = None
        tmp_36 = torch.nn.functional.linear(tmp_35, tmp_3, tmp_2)
        tmp_35 = tmp_3 = tmp_2 = None
        tmp_37 = torch.nn.functional.dropout(tmp_36, 0.1, False, False)
        tmp_36 = None
        tmp_38 = tmp_37.transpose(-2, -1)
        tmp_39 = torch._weight_norm(tmp_7, tmp_6, 2)
        tmp_7 = tmp_6 = None
        tmp_40 = torch.conv1d(tmp_38, tmp_39, tmp_8, (1,), (64,), (1,), 16)
        tmp_38 = tmp_39 = tmp_8 = None
        tmp_41 = tmp_40[Ellipsis, slice(None, -1, None)]
        tmp_40 = None
        tmp_42 = torch.nn.functional.gelu(tmp_41)
        tmp_41 = None
        tmp_43 = tmp_42.transpose(-2, -1)
        tmp_42 = None
        tmp_44 = tmp_37 + tmp_43
        tmp_37 = tmp_43 = None
        tmp_45 = torch.nn.functional.layer_norm(tmp_44, (768,), tmp_5, tmp_4, 1e-05)
        tmp_44 = tmp_5 = tmp_4 = None
        tmp_46 = torch.nn.functional.dropout(tmp_45, 0.1, False, False)
        tmp_45 = None
        return (tmp_46,)