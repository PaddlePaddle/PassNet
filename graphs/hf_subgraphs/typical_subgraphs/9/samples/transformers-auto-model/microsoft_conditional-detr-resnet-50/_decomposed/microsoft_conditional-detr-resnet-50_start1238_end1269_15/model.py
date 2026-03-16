import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, in_0, in_1, in_2, in_3):
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
        tmp_11 = w_11
        tmp_12 = torch.nn.functional.relu(in_1)
        tmp_13 = torch.nn.functional.linear(tmp_12, tmp_11, tmp_10)
        tmp_12 = tmp_11 = tmp_10 = None
        tmp_14 = in_2 * tmp_13
        tmp_13 = None
        tmp_15 = torch.nn.functional.linear(in_0, tmp_5, tmp_4)
        tmp_5 = tmp_4 = None
        tmp_16 = torch.nn.functional.linear(in_3, tmp_7, tmp_6)
        tmp_7 = tmp_6 = None
        tmp_17 = torch.nn.functional.linear(in_0, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_18 = torch.nn.functional.linear(in_3, tmp_3, tmp_2)
        tmp_3 = tmp_2 = None
        tmp_19 = torch.nn.functional.linear(in_0, tmp_9, tmp_8)
        tmp_9 = tmp_8 = None
        tmp_20 = tmp_15 + tmp_16
        tmp_15 = tmp_16 = None
        tmp_21 = tmp_17 + tmp_18
        tmp_17 = tmp_18 = None
        tmp_22 = tmp_20 * 0.1767766952966369
        tmp_20 = None
        tmp_23 = tmp_21.view(1, -1, 8, 32)
        tmp_21 = None
        tmp_24 = tmp_23.transpose(1, 2)
        tmp_23 = None
        tmp_25 = tmp_24.contiguous()
        tmp_24 = None
        tmp_26 = tmp_19.view(1, -1, 8, 32)
        tmp_19 = None
        tmp_27 = tmp_26.transpose(1, 2)
        tmp_26 = None
        tmp_28 = tmp_27.contiguous()
        tmp_27 = None
        tmp_29 = tmp_22.view(1, 300, 8, 32)
        tmp_22 = None
        tmp_30 = tmp_29.transpose(1, 2)
        tmp_29 = None
        tmp_31 = tmp_30.contiguous()
        tmp_30 = None
        tmp_32 = tmp_31.view(8, -1, 32)
        tmp_31 = None
        tmp_33 = tmp_25.view(8, -1, 32)
        tmp_25 = None
        tmp_34 = tmp_28.view(8, -1, 32)
        tmp_28 = None
        tmp_35 = tmp_33.transpose(1, 2)
        tmp_33 = None
        tmp_36 = torch.bmm(tmp_32, tmp_35)
        tmp_32 = tmp_35 = None
        tmp_37 = torch.nn.functional.softmax(tmp_36, dim=-1)
        tmp_36 = None
        tmp_38 = torch.nn.functional.dropout(tmp_37, p=0.0, training=False)
        tmp_37 = None
        tmp_39 = torch.bmm(tmp_38, tmp_34)
        tmp_38 = tmp_34 = None
        tmp_40 = tmp_39.view(1, 8, 300, 32)
        tmp_39 = None
        tmp_41 = tmp_40.transpose(1, 2)
        tmp_40 = None
        tmp_42 = tmp_41.reshape(1, 300, 256)
        tmp_41 = None
        return (tmp_42, tmp_14)