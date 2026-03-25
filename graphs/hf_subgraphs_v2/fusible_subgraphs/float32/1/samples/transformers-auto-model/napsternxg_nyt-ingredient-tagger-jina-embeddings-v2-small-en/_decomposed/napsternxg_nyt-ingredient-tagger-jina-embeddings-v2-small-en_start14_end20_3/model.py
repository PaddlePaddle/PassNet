import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.linear(in_2, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_3 = tmp_2.view((1, 512, 8, 64))
        tmp_2 = None
        tmp_4 = tmp_3.permute(0, 2, 1, 3)
        tmp_3 = None
        tmp_5 = in_4.view((1, 512, 8, 64))
        tmp_6 = tmp_5.permute(0, 2, 1, 3)
        tmp_5 = None
        tmp_7 = in_3.transpose(-1, -2)
        return (tmp_6, tmp_7, tmp_4)