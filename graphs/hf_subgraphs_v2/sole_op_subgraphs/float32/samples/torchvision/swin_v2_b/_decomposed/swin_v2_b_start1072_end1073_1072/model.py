import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        in_0[slice(-8, -4, None), slice(-4, None, None)] = 5
        tmp_0 = in_0
        tmp_0 = None
        return ()