import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = torch.nn.functional.linear(in_3, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_4 = tmp_3.view(1, 12, 199, 2, 4)
        tmp_3 = None
        tmp_5 = tmp_4.sum(-1, keepdim=False)
        tmp_4 = None
        tmp_6 = torch.sigmoid(tmp_5)
        tmp_5 = None
        tmp_7 = tmp_6.chunk(2, dim=-1)
        tmp_6 = None
        tmp_8 = tmp_7[0]
        tmp_9 = tmp_7[1]
        tmp_7 = None
        tmp_10 = tmp_9 * tmp_2
        tmp_9 = tmp_2 = None
        tmp_11 = tmp_10 - 1.0
        tmp_10 = None
        tmp_12 = tmp_8 * tmp_11
        tmp_8 = tmp_11 = None
        tmp_13 = tmp_12 + 2.0
        tmp_12 = None
        tmp_14 = tmp_13.view(1, 12, -1, 1)
        tmp_13 = None
        return (tmp_14,)