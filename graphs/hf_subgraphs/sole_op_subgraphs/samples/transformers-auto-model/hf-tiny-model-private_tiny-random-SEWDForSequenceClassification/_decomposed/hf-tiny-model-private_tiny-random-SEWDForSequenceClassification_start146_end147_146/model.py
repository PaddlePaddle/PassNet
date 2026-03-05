import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.view((1, 3999, 2, -1))
        return (tmp_0,)