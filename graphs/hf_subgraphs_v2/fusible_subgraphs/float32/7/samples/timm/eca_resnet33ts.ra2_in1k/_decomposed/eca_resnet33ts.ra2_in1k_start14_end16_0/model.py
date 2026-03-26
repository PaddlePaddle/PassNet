import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
        tmp_1 = tmp_0.mean((2, 3))
        return (tmp_1, tmp_0)