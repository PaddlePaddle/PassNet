import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.expand(1, 4, 4, 2, 128)
        return (tmp_0,)