import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.nn.functional.linear(in_4, tmp_3, tmp_2)
        tmp_3 = tmp_2 = None
        tmp_5 = tmp_4.view(32, -1, 8, 64)
        tmp_4 = None
        tmp_6 = tmp_5.transpose(1, 2)
        tmp_5 = None
        tmp_7 = torch.nn.functional.linear(in_4, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_8 = tmp_7.view(32, -1, 8, 64)
        tmp_7 = None
        tmp_9 = tmp_8.transpose(1, 2)
        tmp_8 = None
        return (tmp_9, tmp_6)