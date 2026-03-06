import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.expand(1, 1, 19, 19)
        return (tmp_0,)