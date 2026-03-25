import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15):
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
        tmp_17 = tmp_0[slice(None, None, None), None]
        tmp_0 = None
        tmp_18 = torch.conv1d(tmp_17, tmp_6, None, (5,), (0,), (1,), 1)
        tmp_17 = tmp_6 = None
        tmp_19 = torch.nn.functional.group_norm(tmp_18, 512, tmp_8, tmp_7, 1e-05)
        tmp_18 = tmp_8 = tmp_7 = None
        tmp_20 = torch.nn.functional.gelu(tmp_19)
        tmp_19 = None
        tmp_21 = torch.conv1d(tmp_20, tmp_9, None, (2,), (0,), (1,), 1)
        tmp_20 = tmp_9 = None
        tmp_22 = torch.nn.functional.gelu(tmp_21)
        tmp_21 = None
        tmp_23 = torch.conv1d(tmp_22, tmp_10, None, (2,), (0,), (1,), 1)
        tmp_22 = tmp_10 = None
        tmp_24 = torch.nn.functional.gelu(tmp_23)
        tmp_23 = None
        tmp_25 = torch.conv1d(tmp_24, tmp_11, None, (2,), (0,), (1,), 1)
        tmp_24 = tmp_11 = None
        tmp_26 = torch.nn.functional.gelu(tmp_25)
        tmp_25 = None
        tmp_27 = torch.conv1d(tmp_26, tmp_12, None, (2,), (0,), (1,), 1)
        tmp_26 = tmp_12 = None
        tmp_28 = torch.nn.functional.gelu(tmp_27)
        tmp_27 = None
        tmp_29 = torch.conv1d(tmp_28, tmp_13, None, (2,), (0,), (1,), 1)
        tmp_28 = tmp_13 = None
        tmp_30 = torch.nn.functional.gelu(tmp_29)
        tmp_29 = None
        tmp_31 = torch.conv1d(tmp_30, tmp_14, None, (2,), (0,), (1,), 1)
        tmp_30 = tmp_14 = None
        tmp_32 = torch.nn.functional.gelu(tmp_31)
        tmp_31 = None
        tmp_33 = tmp_32.transpose(1, 2)
        tmp_32 = None
        tmp_34 = torch.nn.functional.linear(tmp_33, tmp_16, tmp_15)
        tmp_33 = tmp_16 = tmp_15 = None
        tmp_35 = torch.nn.functional.dropout(tmp_34, 0.0, False, False)
        tmp_34 = None
        tmp_36 = tmp_35.transpose(1, 2)
        tmp_37 = torch._weight_norm(tmp_4, tmp_3, 2)
        tmp_4 = tmp_3 = None
        tmp_38 = torch.conv1d(tmp_36, tmp_37, tmp_5, (1,), (64,), (1,), 16)
        tmp_36 = tmp_37 = tmp_5 = None
        tmp_39 = tmp_38[slice(None, None, None), slice(None, None, None), slice(None, -1, None)]
        tmp_38 = None
        tmp_40 = torch.nn.functional.gelu(tmp_39)
        tmp_39 = None
        tmp_41 = tmp_40.transpose(1, 2)
        tmp_40 = None
        tmp_42 = tmp_35 + tmp_41
        tmp_35 = tmp_41 = None
        tmp_43 = torch.nn.functional.layer_norm(tmp_42, (768,), tmp_2, tmp_1, 1e-05)
        tmp_42 = tmp_2 = tmp_1 = None
        tmp_44 = torch.nn.functional.dropout(tmp_43, 0.1, False, False)
        tmp_43 = None
        tmp_45 = torch.rand([])
        tmp_45 = None
        return (tmp_44,)