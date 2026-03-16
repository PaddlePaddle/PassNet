import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9):
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
        tmp_12 = tmp_1.view(-1, 19)
        tmp_1 = None
        tmp_13 = torch.nn.functional.embedding(tmp_12, tmp_3, 1, None, 2.0, False, False)
        tmp_12 = tmp_3 = None
        tmp_14 = tmp_13 * 32.0
        tmp_13 = None
        tmp_15 = torch.full((19, 19), -65504.0, device=device(type='cuda'))
        tmp_16 = torch.arange(19, device=device(type='cuda'))
        tmp_17 = tmp_16 + 1
        tmp_18 = tmp_17.view(19, 1)
        tmp_17 = None
        tmp_19 = tmp_16 < tmp_18
        tmp_16 = tmp_18 = None
        tmp_20 = tmp_15.masked_fill_(tmp_19, 0)
        tmp_19 = tmp_20 = None
        tmp_21 = tmp_15.to(torch.float16)
        tmp_15 = None
        tmp_22 = tmp_21[None, None, slice(None, None, None), slice(None, None, None)]
        tmp_21 = None
        tmp_23 = tmp_22.expand(1, 1, 19, 19)
        tmp_22 = None
        tmp_24 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_25 = tmp_24.expand(1, 1, 19, 19)
        tmp_24 = None
        tmp_26 = tmp_25.to(torch.float16)
        tmp_25 = None
        tmp_27 = torch.tensor(1.0, dtype=torch.float16)
        tmp_28 = tmp_27 - tmp_26
        tmp_27 = tmp_26 = None
        tmp_29 = tmp_28.to(torch.bool)
        tmp_30 = tmp_28.masked_fill(tmp_29, -65504.0)
        tmp_28 = tmp_29 = None
        tmp_31 = tmp_30.to(device(type='cuda'))
        tmp_30 = None
        tmp_32 = tmp_31.bool()
        tmp_31 = None
        tmp_33 = tmp_23.masked_fill(tmp_32, -65504.0)
        tmp_23 = tmp_32 = None
        tmp_34 = torch.arange(0, 19, dtype=torch.int64, device=device(type='cuda'))
        tmp_35 = tmp_34.unsqueeze(0)
        tmp_34 = None
        tmp_35 += 2
        tmp_36 = tmp_35
        tmp_35 = None
        tmp_37 = tmp_36.view(-1)
        tmp_36 = None
        tmp_38 = tmp_2.index_select(0, tmp_37)
        tmp_2 = tmp_37 = None
        tmp_39 = tmp_38.view(1, 19, 1024)
        tmp_38 = None
        tmp_40 = tmp_39.detach()
        tmp_39 = None
        tmp_41 = tmp_40.to(device(type='cuda'))
        tmp_40 = None
        tmp_42 = tmp_14 + tmp_41
        tmp_14 = tmp_41 = None
        tmp_43 = torch.nn.functional.dropout(tmp_42, p=0.1, training=False)
        tmp_42 = None
        tmp_44 = torch.nn.functional.layer_norm(tmp_43, (1024,), tmp_5, tmp_4, 1e-05)
        tmp_5 = tmp_4 = None
        tmp_45 = torch.nn.functional.linear(tmp_44, tmp_9, tmp_8)
        tmp_9 = tmp_8 = None
        tmp_46 = tmp_45 * 0.125
        tmp_45 = None
        tmp_47 = torch.nn.functional.linear(tmp_44, tmp_7, tmp_6)
        tmp_7 = tmp_6 = None
        tmp_48 = torch.nn.functional.linear(tmp_44, tmp_11, tmp_10)
        tmp_44 = tmp_11 = tmp_10 = None
        tmp_49 = tmp_47.view(1, 19, -1, 64)
        tmp_47 = None
        tmp_50 = tmp_49.transpose(1, 2)
        tmp_49 = None
        tmp_51 = tmp_48.view(1, 19, -1, 64)
        tmp_48 = None
        tmp_52 = tmp_51.transpose(1, 2)
        tmp_51 = None
        tmp_53 = tmp_46.view(1, 19, 16, 64)
        tmp_46 = None
        tmp_54 = tmp_53.transpose(1, 2)
        tmp_53 = None
        tmp_55 = tmp_54.reshape(16, -1, 64)
        tmp_54 = None
        tmp_56 = tmp_50.reshape(16, -1, 64)
        tmp_50 = None
        tmp_57 = tmp_52.reshape(16, -1, 64)
        tmp_52 = None
        tmp_58 = tmp_56.transpose(1, 2)
        tmp_56 = None
        tmp_59 = torch.bmm(tmp_55, tmp_58)
        tmp_55 = tmp_58 = None
        tmp_60 = tmp_59.view(1, 16, 19, 19)
        tmp_59 = None
        tmp_61 = tmp_60 + tmp_33
        tmp_60 = None
        tmp_62 = torch.tensor(-65504.0, device=device(type='cuda'))
        tmp_63 = torch.max(tmp_61, tmp_62)
        tmp_61 = tmp_62 = None
        tmp_64 = tmp_63.view(16, 19, 19)
        tmp_63 = None
        tmp_65 = torch.nn.functional.softmax(tmp_64, dim=-1, dtype=torch.float32)
        tmp_64 = None
        tmp_66 = tmp_65.to(torch.float16)
        tmp_65 = None
        tmp_67 = torch.nn.functional.dropout(tmp_66, p=0.1, training=False)
        tmp_66 = None
        tmp_68 = torch.bmm(tmp_67, tmp_57)
        tmp_67 = tmp_57 = None
        tmp_69 = tmp_68.view(1, 16, 19, 64)
        tmp_68 = None
        tmp_70 = tmp_69.transpose(1, 2)
        tmp_69 = None
        tmp_71 = tmp_70.reshape(1, 19, 1024)
        tmp_70 = None
        return (tmp_71, tmp_33, tmp_43)