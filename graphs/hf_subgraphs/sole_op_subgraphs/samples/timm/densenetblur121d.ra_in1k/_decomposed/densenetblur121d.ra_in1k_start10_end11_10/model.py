import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.pad(in_0, [1, 1, 1, 1], 'reflect', None)
        return (tmp_0,)