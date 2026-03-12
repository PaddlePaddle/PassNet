import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.split([16, 16, 32], dim=3)
        return (tmp_0,)