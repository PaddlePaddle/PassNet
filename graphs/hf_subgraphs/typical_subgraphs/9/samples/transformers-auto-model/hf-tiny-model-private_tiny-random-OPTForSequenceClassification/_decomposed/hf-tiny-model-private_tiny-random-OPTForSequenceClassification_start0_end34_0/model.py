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
        tmp_7 = torch.arange(0, 21, device=device(type='cuda', index=0))
        tmp_8 = torch.full((21, 21), fill_value=-3.4028234663852886e+38, dtype=torch.float32, device=device(type='cuda', index=0))
        tmp_9 = torch.triu(tmp_8, diagonal=1)
        tmp_8 = None
        tmp_10 = torch.arange(21, device=device(type='cuda', index=0))
        tmp_11 = tmp_7.reshape(-1, 1)
        tmp_7 = None
        tmp_12 = tmp_10 > tmp_11
        tmp_10 = tmp_11 = None
        tmp_9 *= tmp_12
        tmp_13 = tmp_9
        tmp_9 = tmp_12 = None
        tmp_14 = tmp_13[None, None, slice(None, None, None), slice(None, None, None)]
        tmp_13 = None
        tmp_15 = tmp_14.expand(1, 1, -1, -1)
        tmp_14 = None
        tmp_16 = tmp_15.clone()
        tmp_15 = None
        tmp_17 = tmp_16[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 21, None)]
        tmp_18 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_19 = tmp_18.to(device(type='cuda', index=0))
        tmp_18 = None
        tmp_20 = tmp_17 + tmp_19
        tmp_17 = tmp_19 = None
        tmp_21 = tmp_20.__eq__(0)
        tmp_20 = None
        tmp_22 = tmp_16[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 21, None)]
        tmp_23 = tmp_22.masked_fill(tmp_21, -3.4028234663852886e+38)
        tmp_22 = tmp_21 = None
        tmp_16[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 21, None)] = tmp_23
        tmp_24 = tmp_16
        tmp_23 = tmp_24 = None
        tmp_25 = tmp_16.__eq__(-3.4028234663852886e+38)
        tmp_26 = torch.all(tmp_25, dim=-1, keepdim=True)
        tmp_25 = None
        tmp_27 = ~tmp_26
        tmp_26 = None
        tmp_28 = tmp_16.mul(tmp_27)
        tmp_16 = tmp_27 = None
        tmp_29 = torch.cumsum(tmp_0, dim=1)
        tmp_30 = tmp_29 * tmp_0
        tmp_29 = tmp_0 = None
        tmp_31 = tmp_30 - 1
        tmp_30 = None
        tmp_32 = tmp_31.long()
        tmp_31 = None
        tmp_33 = tmp_32[slice(None, None, None), slice(0, None, None)]
        tmp_32 = None
        tmp_34 = tmp_33 + 2
        tmp_33 = None
        tmp_35 = torch.nn.functional.embedding(tmp_34, tmp_2, None, None, 2.0, False, False)
        tmp_34 = tmp_2 = None
        tmp_36 = tmp_35.to(device(type='cuda', index=0))
        tmp_35 = None
        tmp_37 = tmp_1 + tmp_36
        tmp_1 = tmp_36 = None
        tmp_38 = torch.nn.functional.layer_norm(tmp_37, (16,), tmp_4, tmp_3, 1e-05)
        tmp_4 = tmp_3 = None
        tmp_39 = torch.nn.functional.linear(tmp_38, tmp_6, tmp_5)
        tmp_6 = tmp_5 = None
        tmp_40 = tmp_39 * 0.5
        tmp_39 = None
        return (tmp_28, tmp_37, tmp_38, tmp_40)