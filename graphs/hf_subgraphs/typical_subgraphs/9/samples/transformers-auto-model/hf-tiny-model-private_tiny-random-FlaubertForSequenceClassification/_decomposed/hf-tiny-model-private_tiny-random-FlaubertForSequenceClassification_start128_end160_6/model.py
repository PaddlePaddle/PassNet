import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_0, in_1, in_2, in_3):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = w_8
        tmp_9 = w_9
        tmp_10 = torch.nn.functional.gelu(in_3)
        tmp_11 = torch.nn.functional.linear(tmp_10, tmp_7, tmp_6)
        tmp_10 = tmp_7 = tmp_6 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, p=0.1, training=False)
        tmp_11 = None
        tmp_13 = in_2 + tmp_12
        tmp_12 = None
        tmp_14 = torch.nn.functional.layer_norm(tmp_13, (32,), tmp_9, tmp_8, 1e-12)
        tmp_13 = tmp_9 = tmp_8 = None
        tmp_15 = in_1.unsqueeze(-1)
        tmp_16 = tmp_15.to(torch.float32)
        tmp_15 = None
        tmp_14 *= tmp_16
        tmp_17 = tmp_14
        tmp_14 = tmp_16 = None
        tmp_18 = torch.nn.functional.linear(tmp_17, tmp_3, tmp_2)
        tmp_3 = tmp_2 = None
        tmp_19 = tmp_18.view(1, -1, 4, 8)
        tmp_18 = None
        tmp_20 = tmp_19.transpose(1, 2)
        tmp_19 = None
        tmp_21 = torch.nn.functional.linear(tmp_17, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_22 = torch.nn.functional.linear(tmp_17, tmp_5, tmp_4)
        tmp_5 = tmp_4 = None
        tmp_23 = tmp_21.view(1, -1, 4, 8)
        tmp_21 = None
        tmp_24 = tmp_23.transpose(1, 2)
        tmp_23 = None
        tmp_25 = tmp_22.view(1, -1, 4, 8)
        tmp_22 = None
        tmp_26 = tmp_25.transpose(1, 2)
        tmp_25 = None
        tmp_27 = tmp_20 / 2.8284271247461903
        tmp_20 = None
        tmp_28 = tmp_24.transpose(2, 3)
        tmp_29 = torch.matmul(tmp_27, tmp_28)
        tmp_27 = tmp_28 = None
        tmp_30 = in_0.__eq__(0)
        tmp_31 = tmp_30.view((1, 1, 1, -1))
        tmp_30 = None
        tmp_32 = tmp_31.expand_as(tmp_29)
        tmp_31 = None
        tmp_33 = tmp_29.masked_fill_(tmp_32, -3.4028234663852886e+38)
        tmp_32 = tmp_33 = None
        tmp_34 = tmp_29.float()
        tmp_35 = torch.nn.functional.softmax(tmp_34, dim=-1)
        tmp_34 = None
        tmp_36 = tmp_35.type_as(tmp_29)
        tmp_35 = tmp_29 = None
        tmp_37 = torch.nn.functional.dropout(tmp_36, p=0.1, training=False)
        tmp_36 = None
        tmp_38 = torch.matmul(tmp_37, tmp_26)
        tmp_37 = None
        tmp_39 = tmp_38.transpose(1, 2)
        tmp_38 = None
        tmp_40 = tmp_39.contiguous()
        tmp_39 = None
        tmp_41 = tmp_40.view(1, -1, 32)
        tmp_40 = None
        return (tmp_41, tmp_24, tmp_17, tmp_26)