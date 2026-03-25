import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.where(in_0, 0.0, -10000000000.0)
        return (tmp_0,)