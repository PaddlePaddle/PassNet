import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        in_0[slice(None, None, None), slice(-4, None, None), slice(-8, -4, None), slice(None, None, None)] = 7
        tmp_0 = in_0
        tmp_0 = None
        return ()