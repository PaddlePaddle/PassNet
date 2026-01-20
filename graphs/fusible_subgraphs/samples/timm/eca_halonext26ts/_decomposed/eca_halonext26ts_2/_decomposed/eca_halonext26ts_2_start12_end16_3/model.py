import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.functional.split(in_0, [16, 64], dim=-1)
        tmp_1 = tmp_0[0]
        tmp_2 = tmp_0[1]
        tmp_0 = None
        tmp_3 = tmp_1.transpose(-1, -2)
        tmp_1 = None
        return (tmp_2, tmp_3)