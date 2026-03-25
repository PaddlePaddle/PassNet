import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
        tmp_1 = tmp_0.split(160, dim=1)
        tmp_0 = None
        tmp_2 = tmp_1[0]
        tmp_3 = tmp_1[1]
        tmp_1 = None
        return (tmp_2, tmp_3)