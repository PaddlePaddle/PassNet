import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = 1e-05 + in_0
        return (tmp_0,)