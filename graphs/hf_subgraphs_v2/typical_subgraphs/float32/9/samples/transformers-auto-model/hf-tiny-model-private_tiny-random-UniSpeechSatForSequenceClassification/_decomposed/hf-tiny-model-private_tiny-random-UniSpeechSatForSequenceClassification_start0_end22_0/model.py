import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13):
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
        tmp_15 = tmp_0[slice(None, None, None), None]
        tmp_0 = None
        tmp_16 = torch.conv1d(tmp_15, tmp_6, None, (4,), (0,), (1,), 1)
        tmp_15 = tmp_6 = None
        tmp_17 = torch.nn.functional.group_norm(tmp_16, 32, tmp_8, tmp_7, 1e-05)
        tmp_16 = tmp_8 = tmp_7 = None
        tmp_18 = torch.nn.functional.gelu(tmp_17)
        tmp_17 = None
        tmp_19 = torch.conv1d(tmp_18, tmp_9, None, (4,), (0,), (1,), 1)
        tmp_18 = tmp_9 = None
        tmp_20 = torch.nn.functional.gelu(tmp_19)
        tmp_19 = None
        tmp_21 = torch.conv1d(tmp_20, tmp_10, None, (4,), (0,), (1,), 1)
        tmp_20 = tmp_10 = None
        tmp_22 = torch.nn.functional.gelu(tmp_21)
        tmp_21 = None
        tmp_23 = tmp_22.transpose(1, 2)
        tmp_22 = None
        tmp_24 = torch.nn.functional.layer_norm(tmp_23, (32,), tmp_12, tmp_11, 1e-05)
        tmp_23 = tmp_12 = tmp_11 = None
        tmp_25 = torch.nn.functional.linear(tmp_24, tmp_14, tmp_13)
        tmp_24 = tmp_14 = tmp_13 = None
        tmp_26 = torch.nn.functional.dropout(tmp_25, 0.0, False, False)
        tmp_25 = None
        tmp_27 = tmp_26.transpose(1, 2)
        tmp_28 = torch._weight_norm(tmp_4, tmp_3, 2)
        tmp_4 = tmp_3 = None
        tmp_29 = torch.conv1d(tmp_27, tmp_28, tmp_5, (1,), (8,), (1,), 2)
        tmp_27 = tmp_28 = tmp_5 = None
        tmp_30 = tmp_29[slice(None, None, None), slice(None, None, None), slice(None, -1, None)]
        tmp_29 = None
        tmp_31 = torch.nn.functional.gelu(tmp_30)
        tmp_30 = None
        tmp_32 = tmp_31.transpose(1, 2)
        tmp_31 = None
        tmp_33 = tmp_26 + tmp_32
        tmp_26 = tmp_32 = None
        tmp_34 = torch.nn.functional.layer_norm(tmp_33, (16,), tmp_2, tmp_1, 1e-05)
        tmp_33 = tmp_2 = tmp_1 = None
        tmp_35 = torch.nn.functional.dropout(tmp_34, 0.1, False, False)
        tmp_34 = None
        tmp_36 = torch.rand([])
        tmp_36 = None
        return (tmp_35,)