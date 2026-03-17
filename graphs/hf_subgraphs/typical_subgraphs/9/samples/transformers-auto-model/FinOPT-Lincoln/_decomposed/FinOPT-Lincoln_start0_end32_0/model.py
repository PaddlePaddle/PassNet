import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3, w_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = w_0
        tmp_3 = w_1
        tmp_4 = w_2
        tmp_5 = w_3
        tmp_6 = w_4
        tmp_7 = tmp_1.view(-1, 20)
        tmp_1 = None
        tmp_8 = torch.nn.functional.embedding(tmp_7, tmp_3, 1, None, 2.0, False, False)
        tmp_7 = tmp_3 = None
        tmp_9 = torch.arange(0, 20, device=device(type='cuda'))
        tmp_10 = torch.full((20, 20), fill_value=-65504.0, dtype=torch.float16, device=device(type='cuda'))
        tmp_11 = torch.triu(tmp_10, diagonal=1)
        tmp_10 = None
        tmp_12 = torch.arange(20, device=device(type='cuda'))
        tmp_13 = tmp_9.reshape(-1, 1)
        tmp_9 = None
        tmp_14 = tmp_12 > tmp_13
        tmp_12 = tmp_13 = None
        tmp_11 *= tmp_14
        tmp_15 = tmp_11
        tmp_11 = tmp_14 = None
        tmp_16 = tmp_15[None, None, slice(None, None, None), slice(None, None, None)]
        tmp_15 = None
        tmp_17 = tmp_16.expand(1, 1, -1, -1)
        tmp_16 = None
        tmp_18 = tmp_17.clone()
        tmp_17 = None
        tmp_19 = tmp_18[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 20, None)]
        tmp_20 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_21 = tmp_20.to(device(type='cuda'))
        tmp_20 = None
        tmp_22 = tmp_19 + tmp_21
        tmp_19 = tmp_21 = None
        tmp_23 = tmp_22 == 0
        tmp_22 = None
        tmp_24 = tmp_18[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 20, None)]
        tmp_25 = tmp_24.masked_fill(tmp_23, -65504.0)
        tmp_24 = tmp_23 = None
        tmp_18[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 20, None)] = tmp_25
        tmp_26 = tmp_18
        tmp_25 = tmp_26 = None
        tmp_27 = torch.cumsum(tmp_0, dim=1)
        tmp_28 = tmp_27 * tmp_0
        tmp_27 = tmp_0 = None
        tmp_29 = tmp_28 - 1
        tmp_28 = None
        tmp_30 = tmp_29.long()
        tmp_29 = None
        tmp_31 = tmp_30[slice(None, None, None), slice(0, None, None)]
        tmp_30 = None
        tmp_32 = tmp_31 + 2
        tmp_31 = None
        tmp_33 = torch.nn.functional.embedding(tmp_32, tmp_2, None, None, 2.0, False, False)
        tmp_32 = tmp_2 = None
        tmp_34 = torch.nn.functional.linear(tmp_8, tmp_6, None)
        tmp_8 = tmp_6 = None
        tmp_35 = tmp_33.to(device(type='cuda'))
        tmp_33 = None
        tmp_36 = tmp_34 + tmp_35
        tmp_34 = tmp_35 = None
        tmp_37 = torch.nn.functional.linear(tmp_36, tmp_5, tmp_4)
        tmp_5 = tmp_4 = None
        tmp_38 = tmp_37 * 0.125
        tmp_37 = None
        return (tmp_18, tmp_36, tmp_38)