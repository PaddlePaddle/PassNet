import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3, w_4, w_5, w_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = w_0
        tmp_3 = w_1
        tmp_4 = w_2
        tmp_5 = w_3
        tmp_6 = w_4
        tmp_7 = w_5
        tmp_8 = w_6
        tmp_9 = torch.nn.functional.embedding(tmp_1, tmp_8, None, None, 2.0, False, False)
        tmp_1 = tmp_8 = None
        tmp_10 = torch.arange(0, 18, device=device(type='cuda'))
        tmp_11 = torch.nn.functional.layer_norm(tmp_9, (1536,), tmp_7, tmp_6, 1e-05)
        tmp_9 = tmp_7 = tmp_6 = None
        tmp_12 = tmp_0.to(device(type='cuda'))
        tmp_0 = None
        tmp_13 = torch.tensor(0.7071067811865476, device=device(type='cuda'), dtype=torch.float32)
        tmp_14 = torch.arange(1, 17, device=device(type='cuda'), dtype=torch.int32)
        tmp_15 = torch.pow(tmp_13, tmp_14)
        tmp_13 = tmp_14 = None
        tmp_16 = tmp_12.cumsum(dim=-1)
        tmp_17 = tmp_16 - 1
        tmp_16 = None
        tmp_18 = tmp_17 * tmp_12
        tmp_17 = None
        tmp_19 = tmp_18[slice(None, None, None), None, slice(None, None, None)]
        tmp_18 = None
        tmp_20 = tmp_15[Ellipsis, None]
        tmp_15 = None
        tmp_21 = tmp_20 * tmp_19
        tmp_20 = tmp_19 = None
        tmp_22 = tmp_21.reshape(16, 1, 18)
        tmp_21 = None
        tmp_23 = tmp_22.to(torch.float16)
        tmp_22 = None
        tmp_24 = torch.full((18, 18), fill_value=-65504.0, dtype=torch.float16, device=device(type='cuda'))
        tmp_25 = torch.triu(tmp_24, diagonal=1)
        tmp_24 = None
        tmp_26 = torch.arange(18, device=device(type='cuda'))
        tmp_27 = tmp_10.reshape(-1, 1)
        tmp_10 = None
        tmp_28 = tmp_26 > tmp_27
        tmp_26 = tmp_27 = None
        tmp_25 *= tmp_28
        tmp_29 = tmp_25
        tmp_25 = tmp_28 = None
        tmp_30 = tmp_29[None, None, slice(None, None, None), slice(None, None, None)]
        tmp_29 = None
        tmp_31 = tmp_30.expand(1, 1, -1, -1)
        tmp_30 = None
        tmp_32 = tmp_31.clone()
        tmp_31 = None
        tmp_33 = tmp_32[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 18, None)]
        tmp_34 = tmp_12[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_12 = None
        tmp_35 = tmp_34.to(device(type='cuda'))
        tmp_34 = None
        tmp_36 = tmp_33 + tmp_35
        tmp_33 = tmp_35 = None
        tmp_37 = tmp_36 == 0
        tmp_36 = None
        tmp_38 = tmp_32[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 18, None)]
        tmp_39 = tmp_38.masked_fill(tmp_37, -65504.0)
        tmp_38 = tmp_37 = None
        tmp_32[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 18, None)] = tmp_39
        tmp_40 = tmp_32
        tmp_39 = tmp_40 = None
        tmp_41 = torch.nn.functional.layer_norm(tmp_11, (1536,), tmp_3, tmp_2, 1e-05)
        tmp_3 = tmp_2 = None
        tmp_42 = torch.nn.functional.linear(tmp_41, tmp_5, tmp_4)
        tmp_41 = tmp_5 = tmp_4 = None
        tmp_43 = tmp_42.view(1, 18, 16, 3, 96)
        tmp_42 = None
        tmp_44 = tmp_43[Ellipsis, 0, slice(None, None, None)]
        tmp_45 = tmp_44.transpose(1, 2)
        tmp_44 = None
        tmp_46 = tmp_43[Ellipsis, 1, slice(None, None, None)]
        tmp_47 = tmp_46.transpose(1, 2)
        tmp_46 = None
        tmp_48 = tmp_43[Ellipsis, 2, slice(None, None, None)]
        tmp_43 = None
        tmp_49 = tmp_48.transpose(1, 2)
        tmp_48 = None
        tmp_50 = tmp_45.reshape(16, -1, 96)
        tmp_45 = None
        tmp_51 = tmp_47.reshape(16, -1, 96)
        tmp_47 = None
        tmp_52 = tmp_51.transpose(-1, -2)
        tmp_51 = None
        tmp_53 = tmp_49.reshape(16, -1, 96)
        tmp_49 = None
        tmp_54 = tmp_23.baddbmm(batch1=tmp_50, batch2=tmp_52, beta=1.0, alpha=0.10206207261596577)
        tmp_50 = tmp_52 = None
        tmp_55 = tmp_54.view(1, 16, 18, -1)
        tmp_54 = None
        tmp_56 = tmp_32[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 18, None)]
        tmp_57 = tmp_55 + tmp_56
        tmp_55 = tmp_56 = None
        tmp_58 = torch.nn.functional.softmax(tmp_57, dim=-1, dtype=torch.float32)
        tmp_57 = None
        tmp_59 = tmp_58.to(torch.float16)
        tmp_58 = None
        tmp_60 = torch.nn.functional.dropout(tmp_59, 0.0, False, False)
        tmp_59 = None
        tmp_61 = tmp_60.view(16, 18, -1)
        tmp_60 = None
        tmp_62 = torch.bmm(tmp_61, tmp_53)
        tmp_61 = tmp_53 = None
        tmp_63 = tmp_62.view(1, 16, 18, 96)
        tmp_62 = None
        tmp_64 = tmp_63.permute(0, 2, 1, 3)
        tmp_63 = None
        tmp_65 = tmp_64.reshape(1, 18, 1536)
        tmp_64 = None
        return (tmp_23, tmp_32, tmp_65, tmp_11)