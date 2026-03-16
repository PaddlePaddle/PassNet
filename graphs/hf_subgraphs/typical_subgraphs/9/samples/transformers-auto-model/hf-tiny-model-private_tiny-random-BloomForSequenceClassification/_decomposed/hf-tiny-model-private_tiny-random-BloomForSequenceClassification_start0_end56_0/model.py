import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3, w_4, w_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = w_0
        tmp_3 = w_1
        tmp_4 = w_2
        tmp_5 = w_3
        tmp_6 = w_4
        tmp_7 = w_5
        tmp_8 = torch.arange(0, 19, device=device(type='cuda', index=0))
        tmp_9 = torch.nn.functional.layer_norm(tmp_1, (32,), tmp_7, tmp_6, 1e-05)
        tmp_1 = tmp_7 = tmp_6 = None
        tmp_10 = tmp_0.to(device(type='cuda', index=0))
        tmp_0 = None
        tmp_11 = torch.tensor(0.25, device=device(type='cuda', index=0), dtype=torch.float32)
        tmp_12 = torch.arange(1, 5, device=device(type='cuda', index=0), dtype=torch.int32)
        tmp_13 = torch.pow(tmp_11, tmp_12)
        tmp_11 = tmp_12 = None
        tmp_14 = tmp_10.cumsum(dim=-1)
        tmp_15 = tmp_14 - 1
        tmp_14 = None
        tmp_16 = tmp_15 * tmp_10
        tmp_15 = None
        tmp_17 = tmp_16[slice(None, None, None), None, slice(None, None, None)]
        tmp_16 = None
        tmp_18 = tmp_13[Ellipsis, None]
        tmp_13 = None
        tmp_19 = tmp_18 * tmp_17
        tmp_18 = tmp_17 = None
        tmp_20 = tmp_19.reshape(4, 1, 19)
        tmp_19 = None
        tmp_21 = tmp_20.to(torch.float32)
        tmp_20 = None
        tmp_22 = torch.full((19, 19), fill_value=-3.4028234663852886e+38, dtype=torch.float32, device=device(type='cuda', index=0))
        tmp_23 = torch.triu(tmp_22, diagonal=1)
        tmp_22 = None
        tmp_24 = torch.arange(19, device=device(type='cuda', index=0))
        tmp_25 = tmp_8.reshape(-1, 1)
        tmp_8 = None
        tmp_26 = tmp_24 > tmp_25
        tmp_24 = tmp_25 = None
        tmp_23 *= tmp_26
        tmp_27 = tmp_23
        tmp_23 = tmp_26 = None
        tmp_28 = tmp_27[None, None, slice(None, None, None), slice(None, None, None)]
        tmp_27 = None
        tmp_29 = tmp_28.expand(1, 1, -1, -1)
        tmp_28 = None
        tmp_30 = tmp_29.clone()
        tmp_29 = None
        tmp_31 = tmp_30[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 19, None)]
        tmp_32 = tmp_10[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_10 = None
        tmp_33 = tmp_32.to(device(type='cuda', index=0))
        tmp_32 = None
        tmp_34 = tmp_31 + tmp_33
        tmp_31 = tmp_33 = None
        tmp_35 = tmp_34.__eq__(0)
        tmp_34 = None
        tmp_36 = tmp_30[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 19, None)]
        tmp_37 = tmp_36.masked_fill(tmp_35, -3.4028234663852886e+38)
        tmp_36 = tmp_35 = None
        tmp_30[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 19, None)] = tmp_37
        tmp_38 = tmp_30
        tmp_37 = tmp_38 = None
        tmp_39 = torch.nn.functional.layer_norm(tmp_9, (32,), tmp_3, tmp_2, 1e-05)
        tmp_3 = tmp_2 = None
        tmp_40 = torch.nn.functional.linear(tmp_39, tmp_5, tmp_4)
        tmp_39 = tmp_5 = tmp_4 = None
        tmp_41 = tmp_40.view(1, 19, 4, 3, 8)
        tmp_40 = None
        tmp_42 = tmp_41[Ellipsis, 0, slice(None, None, None)]
        tmp_43 = tmp_42.transpose(1, 2)
        tmp_42 = None
        tmp_44 = tmp_41[Ellipsis, 1, slice(None, None, None)]
        tmp_45 = tmp_44.transpose(1, 2)
        tmp_44 = None
        tmp_46 = tmp_41[Ellipsis, 2, slice(None, None, None)]
        tmp_41 = None
        tmp_47 = tmp_46.transpose(1, 2)
        tmp_46 = None
        tmp_48 = tmp_43.reshape(4, -1, 8)
        tmp_43 = None
        tmp_49 = tmp_45.reshape(4, -1, 8)
        tmp_50 = tmp_49.transpose(-1, -2)
        tmp_49 = None
        tmp_51 = tmp_47.reshape(4, -1, 8)
        tmp_52 = tmp_21.baddbmm(batch1=tmp_48, batch2=tmp_50, beta=1.0, alpha=0.35355339059327373)
        tmp_48 = tmp_50 = None
        tmp_53 = tmp_52.view(1, 4, 19, -1)
        tmp_52 = None
        tmp_54 = tmp_30[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 19, None)]
        tmp_55 = tmp_53 + tmp_54
        tmp_53 = tmp_54 = None
        tmp_56 = torch.nn.functional.softmax(tmp_55, dim=-1, dtype=torch.float32)
        tmp_55 = None
        tmp_57 = tmp_56.to(torch.float32)
        tmp_56 = None
        tmp_58 = torch.nn.functional.dropout(tmp_57, 0.1, False, False)
        tmp_57 = None
        tmp_59 = tmp_58.view(4, 19, -1)
        tmp_58 = None
        tmp_60 = torch.bmm(tmp_59, tmp_51)
        tmp_59 = tmp_51 = None
        tmp_61 = tmp_60.view(1, 4, 19, 8)
        tmp_60 = None
        tmp_62 = tmp_61.permute(0, 2, 1, 3)
        tmp_61 = None
        tmp_63 = tmp_62.reshape(1, 19, 32)
        tmp_62 = None
        return (tmp_21, tmp_30, tmp_63, tmp_9, tmp_45, tmp_47)