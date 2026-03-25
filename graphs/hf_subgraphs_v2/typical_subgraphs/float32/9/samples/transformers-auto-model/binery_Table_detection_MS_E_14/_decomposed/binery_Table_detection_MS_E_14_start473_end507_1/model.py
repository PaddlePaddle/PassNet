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
        tmp_10 = torch.nn.functional.relu(in_2, inplace=False)
        tmp_11 = torch.nn.functional.dropout(tmp_10, p=0.0, training=False)
        tmp_10 = None
        tmp_12 = torch.nn.functional.linear(tmp_11, tmp_1, tmp_0)
        tmp_11 = tmp_1 = tmp_0 = None
        tmp_13 = torch.nn.functional.dropout(tmp_12, p=0.1, training=False)
        tmp_12 = None
        tmp_14 = in_1 + tmp_13
        tmp_13 = None
        tmp_15 = torch.nn.functional.layer_norm(tmp_14, (256,), tmp_3, tmp_2, 1e-05)
        tmp_14 = tmp_3 = tmp_2 = None
        tmp_16 = tmp_15 + in_3
        tmp_17 = torch.nn.functional.linear(tmp_16, tmp_7, tmp_6)
        tmp_7 = tmp_6 = None
        tmp_18 = tmp_17 * 0.1767766952966369
        tmp_17 = None
        tmp_19 = torch.nn.functional.linear(tmp_16, tmp_5, tmp_4)
        tmp_16 = tmp_5 = tmp_4 = None
        tmp_20 = tmp_19.view(1, -1, 8, 32)
        tmp_19 = None
        tmp_21 = tmp_20.transpose(1, 2)
        tmp_20 = None
        tmp_22 = tmp_21.contiguous()
        tmp_21 = None
        tmp_23 = torch.nn.functional.linear(tmp_15, tmp_9, tmp_8)
        tmp_9 = tmp_8 = None
        tmp_24 = tmp_23.view(1, -1, 8, 32)
        tmp_23 = None
        tmp_25 = tmp_24.transpose(1, 2)
        tmp_24 = None
        tmp_26 = tmp_25.contiguous()
        tmp_25 = None
        tmp_27 = tmp_18.view(1, 625, 8, 32)
        tmp_18 = None
        tmp_28 = tmp_27.transpose(1, 2)
        tmp_27 = None
        tmp_29 = tmp_28.contiguous()
        tmp_28 = None
        tmp_30 = tmp_29.view(8, -1, 32)
        tmp_29 = None
        tmp_31 = tmp_22.view(8, -1, 32)
        tmp_22 = None
        tmp_32 = tmp_26.view(8, -1, 32)
        tmp_26 = None
        tmp_33 = tmp_31.transpose(1, 2)
        tmp_31 = None
        tmp_34 = torch.bmm(tmp_30, tmp_33)
        tmp_30 = tmp_33 = None
        tmp_35 = tmp_34.view(1, 8, 625, 625)
        tmp_34 = None
        tmp_36 = tmp_35 + in_0
        tmp_35 = None
        tmp_37 = tmp_36.view(8, 625, 625)
        tmp_36 = None
        tmp_38 = torch.nn.functional.softmax(tmp_37, dim=-1)
        tmp_37 = None
        tmp_39 = torch.nn.functional.dropout(tmp_38, p=0.0, training=False)
        tmp_38 = None
        tmp_40 = torch.bmm(tmp_39, tmp_32)
        tmp_39 = tmp_32 = None
        tmp_41 = tmp_40.view(1, 8, 625, 32)
        tmp_40 = None
        tmp_42 = tmp_41.transpose(1, 2)
        tmp_41 = None
        tmp_43 = tmp_42.reshape(1, 625, 256)
        tmp_42 = None
        return (tmp_43, tmp_15)