import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, in_0, in_1, in_2, in_3):
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
        tmp_10 = w_10
        tmp_11 = torch.nn.functional.relu(in_2, inplace=False)
        tmp_12 = torch.nn.functional.dropout(tmp_11, p=0.0, training=False)
        tmp_11 = None
        tmp_13 = torch.nn.functional.linear(tmp_12, tmp_1, tmp_0)
        tmp_12 = tmp_1 = tmp_0 = None
        tmp_14 = torch.nn.functional.dropout(tmp_13, p=0.1, training=False)
        tmp_13 = None
        tmp_15 = in_1 + tmp_14
        tmp_14 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (256,), tmp_3, tmp_2, 1e-05)
        tmp_15 = tmp_3 = tmp_2 = None
        tmp_17 = tmp_16 + in_3
        tmp_18 = torch.nn.functional.linear(tmp_17, tmp_7, tmp_6)
        tmp_7 = tmp_6 = None
        tmp_19 = tmp_10.item()
        tmp_10 = None
        tmp_20 = tmp_18 * tmp_19
        tmp_18 = tmp_19 = None
        tmp_21 = torch.nn.functional.linear(tmp_17, tmp_5, tmp_4)
        tmp_17 = tmp_5 = tmp_4 = None
        tmp_22 = tmp_21.view(1, -1, 8, 32)
        tmp_21 = None
        tmp_23 = tmp_22.transpose(1, 2)
        tmp_22 = None
        tmp_24 = tmp_23.contiguous()
        tmp_23 = None
        tmp_25 = torch.nn.functional.linear(tmp_16, tmp_9, tmp_8)
        tmp_9 = tmp_8 = None
        tmp_26 = tmp_25.view(1, -1, 8, 32)
        tmp_25 = None
        tmp_27 = tmp_26.transpose(1, 2)
        tmp_26 = None
        tmp_28 = tmp_27.contiguous()
        tmp_27 = None
        tmp_29 = tmp_20.view(1, 625, 8, 32)
        tmp_20 = None
        tmp_30 = tmp_29.transpose(1, 2)
        tmp_29 = None
        tmp_31 = tmp_30.contiguous()
        tmp_30 = None
        tmp_32 = tmp_31.view(8, -1, 32)
        tmp_31 = None
        tmp_33 = tmp_24.view(8, -1, 32)
        tmp_24 = None
        tmp_34 = tmp_28.view(8, -1, 32)
        tmp_28 = None
        tmp_35 = tmp_33.transpose(1, 2)
        tmp_33 = None
        tmp_36 = torch.bmm(tmp_32, tmp_35)
        tmp_32 = tmp_35 = None
        tmp_37 = tmp_36.view(1, 8, 625, 625)
        tmp_36 = None
        tmp_38 = tmp_37 + in_0
        tmp_37 = None
        tmp_39 = tmp_38.view(8, 625, 625)
        tmp_38 = None
        tmp_40 = torch.nn.functional.softmax(tmp_39, dim=-1)
        tmp_39 = None
        tmp_41 = torch.nn.functional.dropout(tmp_40, p=0.0, training=False)
        tmp_40 = None
        tmp_42 = torch.bmm(tmp_41, tmp_34)
        tmp_41 = tmp_34 = None
        tmp_43 = tmp_42.view(1, 8, 625, 32)
        tmp_42 = None
        tmp_44 = tmp_43.transpose(1, 2)
        tmp_43 = None
        tmp_45 = tmp_44.reshape(1, 625, 256)
        tmp_44 = None
        return (tmp_45, tmp_16)