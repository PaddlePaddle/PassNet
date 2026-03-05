import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.expand(1, 1, 33, 33)
        return (tmp_0,)