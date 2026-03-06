import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.cumsum(in_0, dim=1)
        return (tmp_0,)