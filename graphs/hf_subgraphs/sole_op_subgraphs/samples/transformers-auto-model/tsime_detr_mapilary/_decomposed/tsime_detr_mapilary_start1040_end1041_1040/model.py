import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.split([10000, 2500, 625, 169], dim=1)
        return (tmp_0,)