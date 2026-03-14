import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0, in_1, in_2):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = in_2 * 0.5
        tmp_5 = torch.nn.functional.linear(in_1, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_6 = torch.nn.functional.linear(in_1, tmp_3, tmp_2)
        tmp_3 = tmp_2 = None
        tmp_7 = tmp_5.view(1, -1, 4, 4)
        tmp_5 = None
        tmp_8 = tmp_7.transpose(1, 2)
        tmp_7 = None
        tmp_9 = tmp_6.view(1, -1, 4, 4)
        tmp_6 = None
        tmp_10 = tmp_9.transpose(1, 2)
        tmp_9 = None
        tmp_11 = tmp_4.view(1, 22, 4, 4)
        tmp_4 = None
        tmp_12 = tmp_11.transpose(1, 2)
        tmp_11 = None
        tmp_13 = tmp_12.reshape(4, -1, 4)
        tmp_12 = None
        tmp_14 = tmp_8.reshape(4, -1, 4)
        tmp_15 = tmp_10.reshape(4, -1, 4)
        tmp_16 = tmp_14.transpose(1, 2)
        tmp_14 = None
        tmp_17 = torch.bmm(tmp_13, tmp_16)
        tmp_13 = tmp_16 = None
        tmp_18 = tmp_17.view(1, 4, 22, 22)
        tmp_17 = None
        tmp_19 = tmp_18 + in_0
        tmp_18 = None
        tmp_20 = tmp_19.view(4, 22, 22)
        tmp_19 = None
        tmp_21 = torch.nn.functional.softmax(tmp_20, dim=-1)
        tmp_20 = None
        tmp_22 = torch.nn.functional.dropout(tmp_21, p=0.1, training=False)
        tmp_21 = None
        tmp_23 = torch.bmm(tmp_22, tmp_15)
        tmp_22 = tmp_15 = None
        tmp_24 = tmp_23.view(1, 4, 22, 4)
        tmp_23 = None
        tmp_25 = tmp_24.transpose(1, 2)
        tmp_24 = None
        tmp_26 = tmp_25.reshape(1, 22, 16)
        tmp_25 = None
        return (tmp_26, tmp_8, tmp_10)