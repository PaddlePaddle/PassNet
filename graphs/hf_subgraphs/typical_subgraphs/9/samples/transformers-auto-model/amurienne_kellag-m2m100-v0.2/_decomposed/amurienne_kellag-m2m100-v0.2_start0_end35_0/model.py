import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, w_0, w_1, w_2, w_3, w_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = w_0
        tmp_4 = w_1
        tmp_5 = w_2
        tmp_6 = w_3
        tmp_7 = w_4
        tmp_8 = torch.arange(0, 15, device=device(type='cuda', index=0))
        tmp_9 = torch.full((15, 16), fill_value=-3.4028234663852886e+38, dtype=torch.float32, device=device(type='cuda', index=0))
        tmp_10 = torch.triu(tmp_9, diagonal=1)
        tmp_9 = None
        tmp_11 = torch.arange(16, device=device(type='cuda', index=0))
        tmp_12 = tmp_8.reshape(-1, 1)
        tmp_8 = None
        tmp_13 = tmp_11 > tmp_12
        tmp_11 = tmp_12 = None
        tmp_10 *= tmp_13
        tmp_14 = tmp_10
        tmp_10 = tmp_13 = None
        tmp_15 = tmp_14[None, None, slice(None, None, None), slice(None, None, None)]
        tmp_14 = None
        tmp_16 = tmp_15.expand(1, 1, -1, -1)
        tmp_15 = None
        tmp_17 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_18 = tmp_17.expand(1, 1, 15, 15)
        tmp_17 = None
        tmp_19 = tmp_18.to(torch.float32)
        tmp_18 = None
        tmp_20 = torch.tensor(1.0, dtype=torch.float32)
        tmp_21 = tmp_20 - tmp_19
        tmp_20 = tmp_19 = None
        tmp_22 = tmp_21.to(torch.bool)
        tmp_23 = tmp_21.masked_fill(tmp_22, -3.4028234663852886e+38)
        tmp_21 = tmp_22 = None
        tmp_24 = tmp_1.ne(1)
        tmp_1 = None
        tmp_25 = tmp_24.int()
        tmp_24 = None
        tmp_26 = torch.cumsum(tmp_25, dim=1)
        tmp_27 = tmp_26.type_as(tmp_25)
        tmp_26 = None
        tmp_28 = tmp_27 + 0
        tmp_27 = None
        tmp_29 = tmp_28 * tmp_25
        tmp_28 = tmp_25 = None
        tmp_30 = tmp_29.long()
        tmp_29 = None
        tmp_31 = tmp_30 + 1
        tmp_30 = None
        tmp_32 = tmp_31.to(device(type='cuda', index=0))
        tmp_31 = None
        tmp_33 = tmp_32.view(-1)
        tmp_32 = None
        tmp_34 = tmp_3.index_select(0, tmp_33)
        tmp_3 = tmp_33 = None
        tmp_35 = tmp_34.view(1, 15, 1024)
        tmp_34 = None
        tmp_36 = tmp_35.detach()
        tmp_35 = None
        tmp_37 = tmp_36.to(device(type='cuda', index=0))
        tmp_36 = None
        tmp_38 = tmp_2 + tmp_37
        tmp_2 = tmp_37 = None
        tmp_39 = torch.nn.functional.dropout(tmp_38, p=0.1, training=False)
        tmp_38 = None
        tmp_40 = torch.rand([])
        tmp_40 = None
        tmp_41 = torch.nn.functional.layer_norm(tmp_39, (1024,), tmp_5, tmp_4, 1e-05)
        tmp_5 = tmp_4 = None
        tmp_42 = torch.nn.functional.linear(tmp_41, tmp_7, tmp_6)
        tmp_7 = tmp_6 = None
        return (tmp_16, tmp_23, tmp_39, tmp_41, tmp_42)