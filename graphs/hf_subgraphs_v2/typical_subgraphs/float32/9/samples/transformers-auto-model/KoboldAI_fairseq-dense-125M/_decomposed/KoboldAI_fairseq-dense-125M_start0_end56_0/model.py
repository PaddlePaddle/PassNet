import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8):
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
        tmp_11 = torch.full((9, 9), -3.4028234663852886e+38, device=device(type='cuda', index=0))
        tmp_12 = torch.arange(9, device=device(type='cuda', index=0))
        tmp_13 = tmp_12 + 1
        tmp_14 = tmp_13.view(9, 1)
        tmp_13 = None
        tmp_15 = tmp_12 < tmp_14
        tmp_12 = tmp_14 = None
        tmp_16 = tmp_11.masked_fill_(tmp_15, 0)
        tmp_15 = tmp_16 = None
        tmp_17 = tmp_11.to(torch.float32)
        tmp_11 = None
        tmp_18 = tmp_17[None, None, slice(None, None, None), slice(None, None, None)]
        tmp_17 = None
        tmp_19 = tmp_18.expand(1, 1, 9, 9)
        tmp_18 = None
        tmp_20 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_21 = tmp_20.expand(1, 1, 9, 9)
        tmp_20 = None
        tmp_22 = tmp_21.to(torch.float32)
        tmp_21 = None
        tmp_23 = torch.tensor(1.0, dtype=torch.float32)
        tmp_24 = tmp_23 - tmp_22
        tmp_23 = tmp_22 = None
        tmp_25 = tmp_24.to(torch.bool)
        tmp_26 = tmp_24.masked_fill(tmp_25, -3.4028234663852886e+38)
        tmp_24 = tmp_25 = None
        tmp_27 = tmp_26.to(device(type='cuda', index=0))
        tmp_26 = None
        tmp_28 = tmp_27.bool()
        tmp_27 = None
        tmp_29 = tmp_19.masked_fill(tmp_28, -3.4028234663852886e+38)
        tmp_19 = tmp_28 = None
        tmp_30 = torch.arange(0, 9, dtype=torch.int64, device=device(type='cuda', index=0))
        tmp_31 = tmp_30.unsqueeze(0)
        tmp_30 = None
        tmp_31 += 2
        tmp_32 = tmp_31
        tmp_31 = None
        tmp_33 = tmp_32.view(-1)
        tmp_32 = None
        tmp_34 = tmp_2.index_select(0, tmp_33)
        tmp_2 = tmp_33 = None
        tmp_35 = tmp_34.view(1, 9, 768)
        tmp_34 = None
        tmp_36 = tmp_35.detach()
        tmp_35 = None
        tmp_37 = tmp_36.to(device(type='cuda', index=0))
        tmp_36 = None
        tmp_38 = tmp_1 + tmp_37
        tmp_1 = tmp_37 = None
        tmp_39 = torch.nn.functional.dropout(tmp_38, p=0.1, training=False)
        tmp_38 = None
        tmp_40 = torch.nn.functional.layer_norm(tmp_39, (768,), tmp_4, tmp_3, 1e-05)
        tmp_4 = tmp_3 = None
        tmp_41 = torch.nn.functional.linear(tmp_40, tmp_8, tmp_7)
        tmp_8 = tmp_7 = None
        tmp_42 = tmp_41 * 0.125
        tmp_41 = None
        tmp_43 = torch.nn.functional.linear(tmp_40, tmp_6, tmp_5)
        tmp_6 = tmp_5 = None
        tmp_44 = torch.nn.functional.linear(tmp_40, tmp_10, tmp_9)
        tmp_40 = tmp_10 = tmp_9 = None
        tmp_45 = tmp_43.view(1, 9, -1, 64)
        tmp_43 = None
        tmp_46 = tmp_45.transpose(1, 2)
        tmp_45 = None
        tmp_47 = tmp_44.view(1, 9, -1, 64)
        tmp_44 = None
        tmp_48 = tmp_47.transpose(1, 2)
        tmp_47 = None
        tmp_49 = tmp_42.view(1, 9, 12, 64)
        tmp_42 = None
        tmp_50 = tmp_49.transpose(1, 2)
        tmp_49 = None
        tmp_51 = tmp_50.reshape(12, -1, 64)
        tmp_50 = None
        tmp_52 = tmp_46.reshape(12, -1, 64)
        tmp_53 = tmp_48.reshape(12, -1, 64)
        tmp_54 = tmp_52.transpose(1, 2)
        tmp_52 = None
        tmp_55 = torch.bmm(tmp_51, tmp_54)
        tmp_51 = tmp_54 = None
        tmp_56 = tmp_55.view(1, 12, 9, 9)
        tmp_55 = None
        tmp_57 = tmp_56 + tmp_29
        tmp_56 = None
        tmp_58 = torch.tensor(-3.4028234663852886e+38, device=device(type='cuda', index=0))
        tmp_59 = torch.max(tmp_57, tmp_58)
        tmp_57 = tmp_58 = None
        tmp_60 = tmp_59.view(12, 9, 9)
        tmp_59 = None
        tmp_61 = torch.nn.functional.softmax(tmp_60, dim=-1)
        tmp_60 = None
        tmp_62 = torch.nn.functional.dropout(tmp_61, p=0.1, training=False)
        tmp_61 = None
        tmp_63 = torch.bmm(tmp_62, tmp_53)
        tmp_62 = tmp_53 = None
        tmp_64 = tmp_63.view(1, 12, 9, 64)
        tmp_63 = None
        tmp_65 = tmp_64.transpose(1, 2)
        tmp_64 = None
        tmp_66 = tmp_65.reshape(1, 9, 768)
        tmp_65 = None
        return (tmp_66, tmp_29, tmp_39, tmp_46, tmp_48)