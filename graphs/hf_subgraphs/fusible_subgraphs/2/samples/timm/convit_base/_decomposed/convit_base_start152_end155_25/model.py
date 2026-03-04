import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.linear(in_3, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_3 = tmp_2.permute(0, 3, 1, 2)
        tmp_2 = None
        tmp_4 = in_2.transpose(-2, -1)
        return (tmp_3, tmp_4)