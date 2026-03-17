import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = w_0
        tmp_3 = w_1
        tmp_4 = w_2
        tmp_5 = w_3
        tmp_6 = w_4
        tmp_7 = w_5
        tmp_8 = w_6
        tmp_9 = w_7
        tmp_10 = w_8
        tmp_11 = w_9
        tmp_12 = torch.nn.functional.relu(in_3, inplace=False)
        tmp_13 = torch.nn.functional.dropout(tmp_12, p=0.1, training=False)
        tmp_12 = None
        tmp_14 = torch.nn.functional.linear(tmp_13, tmp_11, tmp_10)
        tmp_13 = tmp_11 = tmp_10 = None
        tmp_15 = torch.nn.functional.dropout(tmp_14, p=0.1, training=False)
        tmp_14 = None
        tmp_16 = in_2 + tmp_15
        tmp_15 = None
        tmp_17 = torch.nn.functional.layer_norm(tmp_16, (1024,), tmp_9, tmp_8, 1e-05)
        tmp_16 = tmp_9 = tmp_8 = None
        tmp_18 = tmp_1.view(-1, 10)
        tmp_18 = None
        tmp_19 = torch.nn.functional.embedding(tmp_1, tmp_7, 0, None, 2.0, False, False)
        tmp_1 = tmp_7 = None
        tmp_20 = tmp_19 * 32.0
        tmp_19 = None
        tmp_21 = torch.arange(0, 10, device=device(type='cuda', index=0))
        tmp_22 = torch.full((10, 11), fill_value=-3.3895313892515355e+38, dtype=torch.bfloat16, device=device(type='cuda', index=0))
        tmp_23 = torch.triu(tmp_22, diagonal=1)
        tmp_22 = None
        tmp_24 = torch.arange(11, device=device(type='cuda', index=0))
        tmp_25 = tmp_21.reshape(-1, 1)
        tmp_26 = tmp_24 > tmp_25
        tmp_24 = tmp_25 = None
        tmp_23 *= tmp_26
        tmp_27 = tmp_23
        tmp_23 = tmp_26 = None
        tmp_28 = tmp_27[None, None, slice(None, None, None), slice(None, None, None)]
        tmp_27 = None
        tmp_29 = tmp_28.expand(1, 1, -1, -1)
        tmp_28 = None
        tmp_30 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_31 = tmp_30.expand(1, 1, 10, 10)
        tmp_30 = None
        tmp_32 = tmp_31.to(torch.bfloat16)
        tmp_31 = None
        tmp_33 = torch.tensor(1.0, dtype=torch.bfloat16)
        tmp_34 = tmp_33 - tmp_32
        tmp_33 = tmp_32 = None
        tmp_35 = tmp_34.to(torch.bool)
        tmp_36 = tmp_34.masked_fill(tmp_35, -3.3895313892515355e+38)
        tmp_34 = tmp_35 = None
        tmp_37 = torch.nn.functional.embedding(tmp_21, tmp_2, None, None, 2.0, False, False)
        tmp_21 = tmp_2 = None
        tmp_38 = tmp_20 + tmp_37
        tmp_20 = tmp_37 = None
        tmp_39 = torch.nn.functional.dropout(tmp_38, p=0.1, training=False)
        tmp_38 = None
        tmp_40 = torch.nn.functional.layer_norm(tmp_39, (1024,), tmp_4, tmp_3, 1e-05)
        tmp_4 = tmp_3 = None
        tmp_41 = torch.nn.functional.linear(tmp_40, tmp_6, tmp_5)
        tmp_6 = tmp_5 = None
        return (tmp_29, tmp_36, tmp_17, tmp_39, tmp_40, tmp_41)