import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, in_0, in_1, in_2, in_3):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = torch.nn.functional.relu(in_2)
        tmp_9 = torch.nn.functional.linear(tmp_8, tmp_7, tmp_6)
        tmp_8 = tmp_7 = tmp_6 = None
        tmp_10 = in_3 * tmp_9
        tmp_9 = None
        tmp_11 = in_1 + tmp_10
        tmp_10 = None
        tmp_12 = torch.nn.functional.linear(tmp_11, tmp_3, tmp_2)
        tmp_3 = tmp_2 = None
        tmp_13 = tmp_12 * 0.1767766952966369
        tmp_12 = None
        tmp_14 = torch.nn.functional.linear(tmp_11, tmp_1, tmp_0)
        tmp_11 = tmp_1 = tmp_0 = None
        tmp_15 = torch.nn.functional.linear(in_1, tmp_5, tmp_4)
        tmp_5 = tmp_4 = None
        tmp_16 = tmp_13.view(1, 625, 8, 32)
        tmp_13 = None
        tmp_17 = tmp_16.transpose(1, 2)
        tmp_16 = None
        tmp_18 = tmp_14.view(1, -1, 8, 32)
        tmp_14 = None
        tmp_19 = tmp_18.transpose(1, 2)
        tmp_18 = None
        tmp_20 = tmp_15.view(1, -1, 8, 32)
        tmp_15 = None
        tmp_21 = tmp_20.transpose(1, 2)
        tmp_20 = None
        tmp_22 = tmp_19.transpose(2, 3)
        tmp_19 = None
        tmp_23 = torch.matmul(tmp_17, tmp_22)
        tmp_17 = tmp_22 = None
        tmp_24 = tmp_23 + in_0
        tmp_23 = None
        tmp_25 = torch.nn.functional.softmax(tmp_24, dim=-1, dtype=torch.float32)
        tmp_24 = None
        tmp_26 = tmp_25.to(torch.float32)
        tmp_25 = None
        tmp_27 = torch.nn.functional.dropout(tmp_26, p=0.0, training=False)
        tmp_26 = None
        tmp_28 = torch.matmul(tmp_27, tmp_21)
        tmp_27 = tmp_21 = None
        tmp_29 = tmp_28.transpose(1, 2)
        tmp_28 = None
        tmp_30 = tmp_29.contiguous()
        tmp_29 = None
        tmp_31 = tmp_30.reshape(1, 625, 256)
        tmp_30 = None
        return (tmp_31,)