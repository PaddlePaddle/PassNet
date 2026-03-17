import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = w_1
        tmp_3 = w_2
        tmp_4 = w_3
        tmp_5 = w_4
        tmp_6 = w_5
        tmp_7 = w_6
        tmp_8 = w_7
        tmp_9 = torch.conv2d(tmp_0, tmp_8, tmp_7, (4, 4), (0, 0), (1, 1), 1)
        tmp_0 = tmp_8 = tmp_7 = None
        tmp_10 = tmp_9.flatten(2)
        tmp_9 = None
        tmp_11 = tmp_10.transpose(1, 2)
        tmp_10 = None
        tmp_12 = torch.nn.functional.layer_norm(tmp_11, (64,), tmp_6, tmp_5, 1e-05)
        tmp_11 = tmp_6 = tmp_5 = None
        tmp_13 = torch.nn.functional.dropout(tmp_12, 0.0, False, False)
        tmp_12 = None
        tmp_14 = torch.nn.functional.layer_norm(tmp_13, (64,), tmp_4, tmp_3, 1e-05)
        tmp_4 = tmp_3 = None
        tmp_15 = tmp_14.view(1, 128, 128, 64)
        tmp_14 = None
        tmp_16 = torch.nn.functional.pad(tmp_15, (0, 0, 0, 5, 0, 5), 'constant', None)
        tmp_15 = None
        tmp_17 = torch.zeros((1, 133, 133), device=device(type='cuda', index=0))
        tmp_18 = tmp_17[slice(None, None, None), slice(-5, None, None), slice(None, None, None)]
        tmp_19 = tmp_18.fill_(1)
        tmp_18 = tmp_19 = None
        tmp_20 = tmp_17[slice(None, None, None), slice(None, None, None), slice(-5, None, None)]
        tmp_21 = tmp_20.fill_(1)
        tmp_20 = tmp_21 = None
        tmp_22 = tmp_16.reshape(1, 19, 7, 19, 7, 64)
        tmp_16 = None
        tmp_23 = tmp_22.transpose(2, 3)
        tmp_22 = None
        tmp_24 = tmp_17.reshape(1, 19, 7, 19, 7)
        tmp_17 = None
        tmp_25 = tmp_24.transpose(2, 3)
        tmp_24 = None
        tmp_26 = tmp_25.reshape(1, 361, 49)
        tmp_25 = None
        tmp_27 = tmp_26.unsqueeze(2)
        tmp_28 = tmp_26.unsqueeze(3)
        tmp_26 = None
        tmp_29 = tmp_27 - tmp_28
        tmp_27 = tmp_28 = None
        tmp_30 = tmp_29 != 0
        tmp_31 = tmp_29.masked_fill(tmp_30, -1000.0)
        tmp_30 = None
        tmp_32 = tmp_29 == 0
        tmp_29 = None
        tmp_33 = tmp_31.masked_fill(tmp_32, 0.0)
        tmp_31 = tmp_32 = None
        tmp_34 = torch.nn.functional.linear(tmp_23, tmp_2, tmp_1)
        tmp_23 = tmp_2 = tmp_1 = None
        tmp_35 = tmp_34.reshape(1, 361, 49, 3, 2, 32)
        tmp_34 = None
        tmp_36 = tmp_35.permute(3, 0, 1, 4, 2, 5)
        tmp_35 = None
        tmp_37 = tmp_36[0]
        tmp_38 = tmp_36[1]
        tmp_39 = tmp_36[2]
        tmp_36 = None
        tmp_40 = tmp_38.transpose(-2, -1)
        tmp_38 = None
        tmp_41 = tmp_37 @ tmp_40
        tmp_37 = tmp_40 = None
        tmp_42 = tmp_41 * 0.1767766952966369
        tmp_41 = None
        tmp_43 = tmp_33.unsqueeze(2)
        tmp_33 = None
        tmp_44 = tmp_42 + tmp_43
        tmp_42 = tmp_43 = None
        tmp_45 = tmp_44.softmax(dim=-1)
        tmp_44 = None
        tmp_46 = torch.nn.functional.dropout(tmp_45, 0.0, False, False)
        tmp_45 = None
        tmp_47 = tmp_46 @ tmp_39
        tmp_46 = tmp_39 = None
        tmp_48 = tmp_47.transpose(2, 3)
        tmp_47 = None
        tmp_49 = tmp_48.reshape(1, 19, 19, 7, 7, 64)
        tmp_48 = None
        tmp_50 = tmp_49.transpose(2, 3)
        tmp_49 = None
        tmp_51 = tmp_50.reshape(1, 133, 133, 64)
        tmp_50 = None
        tmp_52 = tmp_51[slice(None, None, None), slice(None, 128, None), slice(None, 128, None), slice(None, None, None)]
        tmp_51 = None
        tmp_53 = tmp_52.contiguous()
        tmp_52 = None
        tmp_54 = tmp_53.reshape(1, 16384, 64)
        tmp_53 = None
        return (tmp_13, tmp_54)