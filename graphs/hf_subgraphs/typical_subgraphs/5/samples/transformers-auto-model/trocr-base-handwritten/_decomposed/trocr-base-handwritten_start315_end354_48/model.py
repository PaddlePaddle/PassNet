import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18):
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
        tmp_12 = in_12
        tmp_13 = in_13
        tmp_14 = in_14
        tmp_15 = in_15
        tmp_16 = in_16
        tmp_17 = torch.nn.functional.gelu(in_18)
        tmp_18 = torch.nn.functional.linear(tmp_17, tmp_12, tmp_11)
        tmp_17 = tmp_12 = tmp_11 = None
        tmp_19 = torch.nn.functional.dropout(tmp_18, 0.0, False, False)
        tmp_18 = None
        tmp_20 = tmp_19 + in_17
        tmp_19 = None
        tmp_21 = torch.nn.functional.layer_norm(tmp_20, (768,), tmp_14, tmp_13, 1e-12)
        tmp_20 = tmp_14 = tmp_13 = None
        tmp_22 = tmp_21[slice(None, None, None), 0]
        tmp_23 = torch.nn.functional.linear(tmp_22, tmp_16, tmp_15)
        tmp_22 = tmp_16 = tmp_15 = None
        tmp_24 = torch.tanh(tmp_23)
        tmp_23 = tmp_24 = None
        tmp_25 = tmp_0.view(-1, 1)
        tmp_0 = None
        tmp_26 = torch.nn.functional.embedding(tmp_25, tmp_2, 1, None, 2.0, False, False)
        tmp_25 = tmp_2 = None
        tmp_27 = tmp_26 * 1.0
        tmp_26 = None
        tmp_28 = torch.arange(0, 1, dtype=torch.int64, device=device(type='cuda'))
        tmp_29 = tmp_28.expand(1, -1)
        tmp_28 = None
        tmp_30 = tmp_29 + 2
        tmp_29 = None
        tmp_31 = torch.nn.functional.embedding(tmp_30, tmp_1, None, None, 2.0, False, False)
        tmp_30 = tmp_1 = None
        tmp_32 = tmp_27 + tmp_31
        tmp_27 = tmp_31 = None
        tmp_33 = torch.nn.functional.layer_norm(tmp_32, (1024,), tmp_4, tmp_3, 1e-05)
        tmp_32 = tmp_4 = tmp_3 = None
        tmp_34 = torch.nn.functional.dropout(tmp_33, p=0.1, training=False)
        tmp_33 = None
        tmp_35 = torch.nn.functional.linear(tmp_34, tmp_8, tmp_7)
        tmp_8 = tmp_7 = None
        tmp_36 = tmp_35 * 0.125
        tmp_35 = None
        tmp_37 = torch.nn.functional.linear(tmp_34, tmp_6, tmp_5)
        tmp_6 = tmp_5 = None
        tmp_38 = torch.nn.functional.linear(tmp_34, tmp_10, tmp_9)
        tmp_10 = tmp_9 = None
        tmp_39 = tmp_37.view(1, -1, 16, 64)
        tmp_37 = None
        tmp_40 = tmp_39.transpose(1, 2)
        tmp_39 = None
        tmp_41 = tmp_38.view(1, -1, 16, 64)
        tmp_38 = None
        tmp_42 = tmp_41.transpose(1, 2)
        tmp_41 = None
        tmp_43 = tmp_36.view(1, 1, 16, 64)
        tmp_36 = None
        tmp_44 = tmp_43.transpose(1, 2)
        tmp_43 = None
        tmp_45 = tmp_44.reshape(16, -1, 64)
        tmp_44 = None
        tmp_46 = tmp_40.reshape(16, -1, 64)
        tmp_40 = None
        tmp_47 = tmp_42.reshape(16, -1, 64)
        tmp_42 = None
        tmp_48 = tmp_46.transpose(1, 2)
        tmp_46 = None
        tmp_49 = torch.bmm(tmp_45, tmp_48)
        tmp_45 = tmp_48 = None
        tmp_50 = torch.nn.functional.softmax(tmp_49, dim=-1)
        tmp_49 = None
        tmp_51 = torch.nn.functional.dropout(tmp_50, p=0.0, training=False)
        tmp_50 = None
        tmp_52 = torch.bmm(tmp_51, tmp_47)
        tmp_51 = tmp_47 = None
        tmp_53 = tmp_52.view(1, 16, 1, 64)
        tmp_52 = None
        tmp_54 = tmp_53.transpose(1, 2)
        tmp_53 = None
        tmp_55 = tmp_54.reshape(1, 1, 1024)
        tmp_54 = None
        return (tmp_55, tmp_34, tmp_21)