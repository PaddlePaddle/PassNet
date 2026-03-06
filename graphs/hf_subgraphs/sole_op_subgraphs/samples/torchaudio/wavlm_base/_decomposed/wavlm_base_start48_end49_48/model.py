import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.full_like(in_0, 159)
        return (tmp_0,)