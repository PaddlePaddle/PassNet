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
        tmp_7 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_8 = tmp_7.to(dtype=torch.float32)
        tmp_7 = None
        tmp_9 = 1.0 - tmp_8
        tmp_8 = None
        tmp_10 = tmp_9 * -3.4028234663852886e+38
        tmp_9 = None
        tmp_11 = tmp_1.ne(1)
        tmp_12 = tmp_11.int()
        tmp_11 = None
        tmp_13 = torch.cumsum(tmp_12, dim=1)
        tmp_14 = tmp_13.type_as(tmp_12)
        tmp_13 = None
        tmp_15 = tmp_14 * tmp_12
        tmp_14 = tmp_12 = None
        tmp_16 = tmp_15.long()
        tmp_15 = None
        tmp_17 = tmp_16 + 1
        tmp_16 = None
        tmp_18 = torch.nn.functional.embedding(tmp_1, tmp_5, 1, None, 2.0, False, False)
        tmp_1 = tmp_5 = None
        tmp_19 = torch.nn.functional.embedding(tmp_17, tmp_4, 1, None, 2.0, False, False)
        tmp_17 = tmp_4 = None
        tmp_20 = tmp_18 + tmp_19
        tmp_18 = tmp_19 = None
        tmp_21 = torch.nn.functional.layer_norm(tmp_20, (64,), tmp_3, tmp_2, 1e-12)
        tmp_20 = tmp_3 = tmp_2 = None
        tmp_22 = torch.nn.functional.dropout(tmp_21, 0.1, False, False)
        tmp_21 = None
        tmp_23 = torch.arange(45, dtype=torch.int64)
        tmp_24 = tmp_23[slice(None, None, None), None]
        tmp_23 = None
        tmp_25 = torch.arange(45, dtype=torch.int64)
        tmp_26 = tmp_25[None, slice(None, None, None)]
        tmp_25 = None
        tmp_27 = tmp_26 - tmp_24
        tmp_26 = tmp_24 = None
        tmp_28 = -tmp_27
        tmp_27 = None
        tmp_29 = tmp_28 < 0
        tmp_30 = tmp_29.to(torch.int64)
        tmp_29 = None
        tmp_31 = tmp_30 * 16
        tmp_30 = None
        tmp_32 = 0 + tmp_31
        tmp_31 = None
        tmp_33 = torch.abs(tmp_28)
        tmp_28 = None
        tmp_34 = tmp_33 < 8
        tmp_35 = tmp_33.float()
        tmp_36 = tmp_35 / 8
        tmp_35 = None
        tmp_37 = torch.log(tmp_36)
        tmp_36 = None
        tmp_38 = tmp_37 / 2.772588722239781
        tmp_37 = None
        tmp_39 = tmp_38 * 8
        tmp_38 = None
        tmp_40 = tmp_39.to(torch.int64)
        tmp_39 = None
        tmp_41 = 8 + tmp_40
        tmp_40 = None
        tmp_42 = torch.full_like(tmp_41, 15)
        tmp_43 = torch.min(tmp_41, tmp_42)
        tmp_41 = tmp_42 = None
        tmp_44 = torch.where(tmp_34, tmp_33, tmp_43)
        tmp_34 = tmp_33 = tmp_43 = None
        tmp_32 += tmp_44
        tmp_45 = tmp_32
        tmp_32 = tmp_44 = None
        tmp_46 = tmp_45.to(device(type='cuda', index=0))
        tmp_45 = None
        tmp_47 = torch.nn.functional.embedding(tmp_46, tmp_6, None, None, 2.0, False, False)
        tmp_46 = tmp_6 = None
        tmp_48 = tmp_47.permute([2, 0, 1])
        tmp_47 = None
        tmp_49 = tmp_48.unsqueeze(0)
        tmp_48 = None
        tmp_50 = tmp_49.expand((1, -1, 45, 45))
        tmp_49 = None
        tmp_51 = tmp_50.contiguous()
        tmp_50 = None
        return (tmp_22, tmp_10, tmp_51)