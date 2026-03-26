import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = 1.0 + in_0
        return (tmp_0,)