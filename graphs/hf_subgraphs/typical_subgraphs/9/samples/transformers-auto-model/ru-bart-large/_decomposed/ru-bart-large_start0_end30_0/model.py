import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7):
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
        tmp_10 = tmp_1.view(-1, 22)
        tmp_10 = None
        tmp_11 = torch.nn.functional.embedding(tmp_1, tmp_3, 1, None, 2.0, False, False)
        tmp_1 = tmp_3 = None
        tmp_12 = tmp_11 * 32.0
        tmp_11 = None
        tmp_13 = torch.arange(0, 22, device=device(type='cuda'))
        tmp_14 = torch.full((22, 22), fill_value=-65504.0, dtype=torch.float16, device=device(type='cuda'))
        tmp_15 = torch.triu(tmp_14, diagonal=1)
        tmp_14 = None
        tmp_16 = torch.arange(22, device=device(type='cuda'))
        tmp_17 = tmp_13.reshape(-1, 1)
        tmp_18 = tmp_16 > tmp_17
        tmp_16 = tmp_17 = None
        tmp_15 *= tmp_18
        tmp_19 = tmp_15
        tmp_15 = tmp_18 = None
        tmp_20 = tmp_19[None, None, slice(None, None, None), slice(None, None, None)]
        tmp_19 = None
        tmp_21 = tmp_20.expand(1, 1, -1, -1)
        tmp_20 = None
        tmp_22 = tmp_21.clone()
        tmp_21 = None
        tmp_23 = tmp_22[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 22, None)]
        tmp_24 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_25 = tmp_24.to(device(type='cuda'))
        tmp_24 = None
        tmp_26 = tmp_23 + tmp_25
        tmp_23 = tmp_25 = None
        tmp_27 = tmp_26 == 0
        tmp_26 = None
        tmp_28 = tmp_22[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 22, None)]
        tmp_29 = tmp_28.masked_fill(tmp_27, -65504.0)
        tmp_28 = tmp_27 = None
        tmp_22[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 22, None)] = tmp_29
        tmp_30 = tmp_22
        tmp_29 = tmp_30 = None
        tmp_31 = tmp_13.unsqueeze(0)
        tmp_13 = None
        tmp_32 = tmp_31 + 2
        tmp_31 = None
        tmp_33 = torch.nn.functional.embedding(tmp_32, tmp_2, None, None, 2.0, False, False)
        tmp_32 = tmp_2 = None
        tmp_34 = tmp_33.to(device(type='cuda'))
        tmp_33 = None
        tmp_35 = tmp_12 + tmp_34
        tmp_12 = tmp_34 = None
        tmp_36 = torch.nn.functional.layer_norm(tmp_35, (1024,), tmp_5, tmp_4, 1e-05)
        tmp_35 = tmp_5 = tmp_4 = None
        tmp_37 = torch.nn.functional.dropout(tmp_36, p=0.1, training=False)
        tmp_36 = None
        tmp_38 = torch.nn.functional.layer_norm(tmp_37, (1024,), tmp_7, tmp_6, 1e-05)
        tmp_7 = tmp_6 = None
        tmp_39 = torch.nn.functional.linear(tmp_38, tmp_9, tmp_8)
        tmp_9 = tmp_8 = None
        return (tmp_22, tmp_37, tmp_38, tmp_39)