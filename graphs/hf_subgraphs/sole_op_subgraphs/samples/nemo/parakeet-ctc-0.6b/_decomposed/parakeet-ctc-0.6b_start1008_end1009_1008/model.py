import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_1.masked_fill(in_0, -10000.0)
        return (tmp_0,)