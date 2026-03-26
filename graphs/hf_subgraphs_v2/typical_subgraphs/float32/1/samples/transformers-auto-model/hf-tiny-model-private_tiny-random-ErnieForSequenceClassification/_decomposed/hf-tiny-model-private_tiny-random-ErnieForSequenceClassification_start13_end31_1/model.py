import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_6.view(1, -1, 4, 8)
        tmp_5 = tmp_4.transpose(1, 2)
        tmp_4 = None
        tmp_6 = torch.nn.functional.linear(in_4, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_7 = tmp_6.view(1, -1, 4, 8)
        tmp_6 = None
        tmp_8 = tmp_7.transpose(1, 2)
        tmp_7 = None
        tmp_9 = torch.nn.functional.linear(in_4, tmp_3, tmp_2)
        tmp_3 = tmp_2 = None
        tmp_10 = tmp_9.view(1, -1, 4, 8)
        tmp_9 = None
        tmp_11 = tmp_10.transpose(1, 2)
        tmp_10 = None
        tmp_12 = tmp_8.transpose(-1, -2)
        tmp_8 = None
        tmp_13 = torch.matmul(tmp_5, tmp_12)
        tmp_5 = tmp_12 = None
        tmp_14 = tmp_13 / 2.8284271247461903
        tmp_13 = None
        tmp_15 = tmp_14 + in_5
        tmp_14 = None
        tmp_16 = torch.nn.functional.softmax(tmp_15, dim=-1)
        tmp_15 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, 0.1, False, False)
        tmp_16 = None
        tmp_18 = torch.matmul(tmp_17, tmp_11)
        tmp_17 = tmp_11 = None
        tmp_19 = tmp_18.permute(0, 2, 1, 3)
        tmp_18 = None
        tmp_20 = tmp_19.contiguous()
        tmp_19 = None
        tmp_21 = tmp_20.view((1, 512, 32))
        tmp_20 = None
        return (tmp_21,)