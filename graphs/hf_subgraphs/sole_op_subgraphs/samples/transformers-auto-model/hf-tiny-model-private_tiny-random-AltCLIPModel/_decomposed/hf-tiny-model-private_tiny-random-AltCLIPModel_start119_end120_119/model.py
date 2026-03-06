import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0 / 2.8284271247461903
        return (tmp_0,)