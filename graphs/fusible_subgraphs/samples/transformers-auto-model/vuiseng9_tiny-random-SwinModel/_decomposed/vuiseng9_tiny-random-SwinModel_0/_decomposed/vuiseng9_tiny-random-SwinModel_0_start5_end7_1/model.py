import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0):
        tmp_0 = in_0 / 2.8284271247461903
        tmp_1 = w_0.view(-1)
        return (tmp_0, tmp_1)