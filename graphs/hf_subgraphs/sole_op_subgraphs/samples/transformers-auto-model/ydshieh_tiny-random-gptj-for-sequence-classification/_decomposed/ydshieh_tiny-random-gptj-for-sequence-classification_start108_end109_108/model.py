import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.repeat(1, 1, 8)
        return (tmp_0,)