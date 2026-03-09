import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.where(in_0, 1.0, -1000.0)
        return (tmp_0,)