import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.chunk(in_0, 2, dim=-1)
        return (tmp_0,)