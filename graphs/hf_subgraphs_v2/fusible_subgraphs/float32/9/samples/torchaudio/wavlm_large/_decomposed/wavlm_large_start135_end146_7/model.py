import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.linear(in_2, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_3 = tmp_2.chunk(3, -1)
        tmp_2 = None
        tmp_4 = tmp_3[0]
        tmp_5 = tmp_3[1]
        tmp_6 = tmp_3[2]
        tmp_3 = None
        tmp_7 = tmp_4.view((1, 199, 16, 64))
        tmp_4 = None
        tmp_8 = tmp_7.transpose(2, 1)
        tmp_7 = None
        tmp_9 = tmp_5.view((1, 199, 16, 64))
        tmp_5 = None
        tmp_10 = tmp_9.transpose(2, 1)
        tmp_9 = None
        tmp_11 = tmp_6.view((1, 199, 16, 64))
        tmp_6 = None
        tmp_12 = tmp_11.transpose(2, 1)
        tmp_11 = None
        return (tmp_10, tmp_8, tmp_12)