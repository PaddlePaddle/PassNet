import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        in_0[slice(-4, None, None), slice(-8, -4, None)] = 7
        tmp_0 = in_0
        tmp_0 = None
        return ()