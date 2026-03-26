import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = w_8
        tmp_9 = torch.nn.functional.gelu(in_1, approximate='none')
        tmp_10 = torch.nn.functional.dropout(tmp_9, 0.0, False, False)
        tmp_9 = None
        tmp_11 = torch.nn.functional.linear(tmp_10, tmp_1, tmp_0)
        tmp_10 = tmp_1 = tmp_0 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, 0.0, False, False)
        tmp_11 = None
        tmp_13 = in_0 + tmp_12
        tmp_12 = None
        tmp_14 = torch.nn.functional.layer_norm(tmp_13, (768,), tmp_8, tmp_7, 1e-06)
        tmp_8 = tmp_7 = None
        tmp_15 = torch.zeros(1, 196, 196, 3)
        tmp_16 = torch.arange(14)
        tmp_17 = tmp_16.view(1, -1)
        tmp_16 = None
        tmp_18 = torch.arange(14)
        tmp_19 = tmp_18.view(-1, 1)
        tmp_18 = None
        tmp_20 = tmp_17 - tmp_19
        tmp_17 = tmp_19 = None
        tmp_21 = tmp_20.repeat(14, 14)
        tmp_22 = tmp_20.repeat_interleave(14, dim=0)
        tmp_20 = None
        tmp_23 = tmp_22.repeat_interleave(14, dim=1)
        tmp_22 = None
        tmp_24 = tmp_21 ** 2
        tmp_25 = tmp_23 ** 2
        tmp_26 = tmp_24 + tmp_25
        tmp_24 = tmp_25 = None
        tmp_27 = tmp_26.unsqueeze(0)
        tmp_26 = None
        tmp_15[slice(None, None, None), slice(None, None, None), slice(None, None, None), 2] = tmp_27
        tmp_28 = tmp_15
        tmp_27 = tmp_28 = None
        tmp_29 = tmp_23.unsqueeze(0)
        tmp_23 = None
        tmp_15[slice(None, None, None), slice(None, None, None), slice(None, None, None), 1] = tmp_29
        tmp_30 = tmp_15
        tmp_29 = tmp_30 = None
        tmp_31 = tmp_21.unsqueeze(0)
        tmp_21 = None
        tmp_15[slice(None, None, None), slice(None, None, None), slice(None, None, None), 0] = tmp_31
        tmp_32 = tmp_15
        tmp_31 = tmp_32 = None
        tmp_33 = tmp_15.to(device(type='cuda'))
        tmp_15 = None
        tmp_34 = torch.nn.functional.linear(tmp_14, tmp_4, None)
        tmp_4 = None
        tmp_35 = tmp_34.reshape(1, 196, 2, 16, 48)
        tmp_34 = None
        tmp_36 = tmp_35.permute(2, 0, 3, 1, 4)
        tmp_35 = None
        tmp_37 = tmp_36[0]
        tmp_38 = tmp_36[1]
        tmp_36 = None
        tmp_39 = tmp_33.expand(1, -1, -1, -1)
        tmp_40 = torch.nn.functional.linear(tmp_39, tmp_3, tmp_2)
        tmp_39 = tmp_3 = tmp_2 = None
        tmp_41 = tmp_40.permute(0, 3, 1, 2)
        tmp_40 = None
        tmp_42 = tmp_38.transpose(-2, -1)
        tmp_38 = None
        tmp_43 = tmp_37 @ tmp_42
        tmp_37 = tmp_42 = None
        tmp_44 = tmp_43 * 0.14433756729740643
        tmp_43 = None
        tmp_45 = tmp_44.softmax(dim=-1)
        tmp_44 = None
        tmp_46 = tmp_41.softmax(dim=-1)
        tmp_41 = None
        tmp_47 = tmp_6.view(1, -1, 1, 1)
        tmp_6 = None
        tmp_48 = torch.sigmoid(tmp_47)
        tmp_49 = 1.0 - tmp_48
        tmp_48 = None
        tmp_50 = tmp_49 * tmp_45
        tmp_49 = tmp_45 = None
        tmp_51 = torch.sigmoid(tmp_47)
        tmp_47 = None
        tmp_52 = tmp_51 * tmp_46
        tmp_51 = tmp_46 = None
        tmp_53 = tmp_50 + tmp_52
        tmp_50 = tmp_52 = None
        tmp_54 = tmp_53.sum(dim=-1)
        tmp_55 = tmp_54.unsqueeze(-1)
        tmp_54 = None
        tmp_53 /= tmp_55
        tmp_56 = tmp_53
        tmp_53 = tmp_55 = None
        tmp_57 = torch.nn.functional.dropout(tmp_56, 0.0, False, False)
        tmp_56 = None
        tmp_58 = torch.nn.functional.linear(tmp_14, tmp_5, None)
        tmp_14 = tmp_5 = None
        tmp_59 = tmp_58.reshape(1, 196, 16, 48)
        tmp_58 = None
        tmp_60 = tmp_59.permute(0, 2, 1, 3)
        tmp_59 = None
        tmp_61 = tmp_57 @ tmp_60
        tmp_57 = tmp_60 = None
        tmp_62 = tmp_61.transpose(1, 2)
        tmp_61 = None
        tmp_63 = tmp_62.reshape(1, 196, 768)
        tmp_62 = None
        return (tmp_33, tmp_63, tmp_13)