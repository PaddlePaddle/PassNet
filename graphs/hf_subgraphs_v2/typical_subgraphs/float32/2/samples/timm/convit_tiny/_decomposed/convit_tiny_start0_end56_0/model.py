import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = in_7
        tmp_8 = in_8
        tmp_9 = in_9
        tmp_10 = in_10
        tmp_11 = in_11
        tmp_12 = torch.conv2d(tmp_11, tmp_8, tmp_7, (16, 16), (0, 0), (1, 1), 1)
        tmp_11 = tmp_8 = tmp_7 = None
        tmp_13 = tmp_12.flatten(2)
        tmp_12 = None
        tmp_14 = tmp_13.transpose(1, 2)
        tmp_13 = None
        tmp_15 = tmp_14 + tmp_10
        tmp_14 = tmp_10 = None
        tmp_16 = torch.nn.functional.dropout(tmp_15, 0.0, False, False)
        tmp_15 = None
        tmp_17 = tmp_9.expand(1, -1, -1)
        tmp_9 = None
        tmp_18 = torch.nn.functional.layer_norm(tmp_16, (192,), tmp_6, tmp_5, 1e-06)
        tmp_6 = tmp_5 = None
        tmp_19 = torch.zeros(1, 196, 196, 3)
        tmp_20 = torch.arange(14)
        tmp_21 = tmp_20.view(1, -1)
        tmp_20 = None
        tmp_22 = torch.arange(14)
        tmp_23 = tmp_22.view(-1, 1)
        tmp_22 = None
        tmp_24 = tmp_21 - tmp_23
        tmp_21 = tmp_23 = None
        tmp_25 = tmp_24.repeat(14, 14)
        tmp_26 = tmp_24.repeat_interleave(14, dim=0)
        tmp_24 = None
        tmp_27 = tmp_26.repeat_interleave(14, dim=1)
        tmp_26 = None
        tmp_28 = tmp_25 ** 2
        tmp_29 = tmp_27 ** 2
        tmp_30 = tmp_28 + tmp_29
        tmp_28 = tmp_29 = None
        tmp_31 = tmp_30.unsqueeze(0)
        tmp_30 = None
        tmp_19[slice(None, None, None), slice(None, None, None), slice(None, None, None), 2] = tmp_31
        tmp_32 = tmp_19
        tmp_31 = tmp_32 = None
        tmp_33 = tmp_27.unsqueeze(0)
        tmp_27 = None
        tmp_19[slice(None, None, None), slice(None, None, None), slice(None, None, None), 1] = tmp_33
        tmp_34 = tmp_19
        tmp_33 = tmp_34 = None
        tmp_35 = tmp_25.unsqueeze(0)
        tmp_25 = None
        tmp_19[slice(None, None, None), slice(None, None, None), slice(None, None, None), 0] = tmp_35
        tmp_36 = tmp_19
        tmp_35 = tmp_36 = None
        tmp_37 = tmp_19.to(device(type='cuda'))
        tmp_19 = None
        tmp_38 = torch.nn.functional.linear(tmp_18, tmp_2, None)
        tmp_2 = None
        tmp_39 = tmp_38.reshape(1, 196, 2, 4, 48)
        tmp_38 = None
        tmp_40 = tmp_39.permute(2, 0, 3, 1, 4)
        tmp_39 = None
        tmp_41 = tmp_40[0]
        tmp_42 = tmp_40[1]
        tmp_40 = None
        tmp_43 = tmp_37.expand(1, -1, -1, -1)
        tmp_44 = torch.nn.functional.linear(tmp_43, tmp_1, tmp_0)
        tmp_43 = tmp_1 = tmp_0 = None
        tmp_45 = tmp_44.permute(0, 3, 1, 2)
        tmp_44 = None
        tmp_46 = tmp_42.transpose(-2, -1)
        tmp_42 = None
        tmp_47 = tmp_41 @ tmp_46
        tmp_41 = tmp_46 = None
        tmp_48 = tmp_47 * 0.14433756729740643
        tmp_47 = None
        tmp_49 = tmp_48.softmax(dim=-1)
        tmp_48 = None
        tmp_50 = tmp_45.softmax(dim=-1)
        tmp_45 = None
        tmp_51 = tmp_4.view(1, -1, 1, 1)
        tmp_4 = None
        tmp_52 = torch.sigmoid(tmp_51)
        tmp_53 = 1.0 - tmp_52
        tmp_52 = None
        tmp_54 = tmp_53 * tmp_49
        tmp_53 = tmp_49 = None
        tmp_55 = torch.sigmoid(tmp_51)
        tmp_51 = None
        tmp_56 = tmp_55 * tmp_50
        tmp_55 = tmp_50 = None
        tmp_57 = tmp_54 + tmp_56
        tmp_54 = tmp_56 = None
        tmp_58 = tmp_57.sum(dim=-1)
        tmp_59 = tmp_58.unsqueeze(-1)
        tmp_58 = None
        tmp_57 /= tmp_59
        tmp_60 = tmp_57
        tmp_57 = tmp_59 = None
        tmp_61 = torch.nn.functional.dropout(tmp_60, 0.0, False, False)
        tmp_60 = None
        tmp_62 = torch.nn.functional.linear(tmp_18, tmp_3, None)
        tmp_18 = tmp_3 = None
        tmp_63 = tmp_62.reshape(1, 196, 4, 48)
        tmp_62 = None
        tmp_64 = tmp_63.permute(0, 2, 1, 3)
        tmp_63 = None
        tmp_65 = tmp_61 @ tmp_64
        tmp_61 = tmp_64 = None
        tmp_66 = tmp_65.transpose(1, 2)
        tmp_65 = None
        tmp_67 = tmp_66.reshape(1, 196, 192)
        tmp_66 = None
        return (tmp_17, tmp_37, tmp_16, tmp_67)