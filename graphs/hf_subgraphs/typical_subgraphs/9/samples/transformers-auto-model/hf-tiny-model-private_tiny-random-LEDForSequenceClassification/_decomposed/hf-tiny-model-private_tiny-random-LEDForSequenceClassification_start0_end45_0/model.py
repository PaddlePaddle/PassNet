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
        tmp_11 = torch.full((22, 22), -3.4028234663852886e+38, device=device(type='cuda', index=0))
        tmp_12 = torch.arange(22, device=device(type='cuda', index=0))
        tmp_13 = tmp_12 + 1
        tmp_14 = tmp_13.view(22, 1)
        tmp_13 = None
        tmp_15 = tmp_12 < tmp_14
        tmp_12 = tmp_14 = None
        tmp_16 = tmp_11.masked_fill_(tmp_15, 0)
        tmp_15 = tmp_16 = None
        tmp_17 = tmp_11.to(torch.float32)
        tmp_11 = None
        tmp_18 = tmp_17[None, None, slice(None, None, None), slice(None, None, None)]
        tmp_17 = None
        tmp_19 = tmp_18.expand(1, 1, 22, 22)
        tmp_18 = None
        tmp_20 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_21 = tmp_20.expand(1, 1, 22, 22)
        tmp_20 = None
        tmp_22 = tmp_21.to(torch.float32)
        tmp_21 = None
        tmp_23 = 1.0 - tmp_22
        tmp_22 = None
        tmp_24 = tmp_23.bool()
        tmp_25 = tmp_23.masked_fill(tmp_24, -3.4028234663852886e+38)
        tmp_24 = None
        tmp_26 = tmp_25 * tmp_23
        tmp_25 = tmp_23 = None
        tmp_27 = torch.arange(0, 22, dtype=torch.int64, device=device(type='cuda', index=0))
        tmp_28 = torch.nn.functional.embedding(tmp_27, tmp_2, None, None, 2.0, False, False)
        tmp_27 = tmp_2 = None
        tmp_29 = tmp_1 + tmp_28
        tmp_1 = tmp_28 = None
        tmp_30 = torch.nn.functional.layer_norm(tmp_29, (16,), tmp_4, tmp_3, 1e-05)
        tmp_29 = tmp_4 = tmp_3 = None
        tmp_31 = torch.nn.functional.dropout(tmp_30, p=0.1, training=False)
        tmp_30 = None
        tmp_32 = torch.nn.functional.linear(tmp_31, tmp_8, tmp_7)
        tmp_8 = tmp_7 = None
        tmp_33 = tmp_32 * 0.5
        tmp_32 = None
        tmp_34 = torch.nn.functional.linear(tmp_31, tmp_6, tmp_5)
        tmp_6 = tmp_5 = None
        tmp_35 = torch.nn.functional.linear(tmp_31, tmp_10, tmp_9)
        tmp_10 = tmp_9 = None
        tmp_36 = tmp_34.view(1, -1, 4, 4)
        tmp_34 = None
        tmp_37 = tmp_36.transpose(1, 2)
        tmp_36 = None
        tmp_38 = tmp_35.view(1, -1, 4, 4)
        tmp_35 = None
        tmp_39 = tmp_38.transpose(1, 2)
        tmp_38 = None
        tmp_40 = tmp_33.view(1, 22, 4, 4)
        tmp_33 = None
        tmp_41 = tmp_40.transpose(1, 2)
        tmp_40 = None
        tmp_42 = tmp_41.reshape(4, -1, 4)
        tmp_41 = None
        tmp_43 = tmp_37.reshape(4, -1, 4)
        tmp_44 = tmp_39.reshape(4, -1, 4)
        tmp_45 = tmp_43.transpose(1, 2)
        tmp_43 = None
        tmp_46 = torch.bmm(tmp_42, tmp_45)
        tmp_42 = tmp_45 = None
        tmp_47 = tmp_46.view(1, 4, 22, 22)
        tmp_46 = None
        tmp_48 = tmp_47 + tmp_19
        tmp_47 = None
        tmp_49 = tmp_48.view(4, 22, 22)
        tmp_48 = None
        tmp_50 = torch.nn.functional.softmax(tmp_49, dim=-1)
        tmp_49 = None
        tmp_51 = torch.nn.functional.dropout(tmp_50, p=0.1, training=False)
        tmp_50 = None
        tmp_52 = torch.bmm(tmp_51, tmp_44)
        tmp_51 = tmp_44 = None
        tmp_53 = tmp_52.view(1, 4, 22, 4)
        tmp_52 = None
        tmp_54 = tmp_53.transpose(1, 2)
        tmp_53 = None
        tmp_55 = tmp_54.reshape(1, 22, 16)
        tmp_54 = None
        return (tmp_55, tmp_19, tmp_26, tmp_31, tmp_37, tmp_39)