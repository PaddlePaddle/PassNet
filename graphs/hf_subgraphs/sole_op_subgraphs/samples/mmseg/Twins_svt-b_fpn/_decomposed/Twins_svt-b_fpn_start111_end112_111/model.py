import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0.masked_fill(in_1, -1000.0)
        return (tmp_0,)