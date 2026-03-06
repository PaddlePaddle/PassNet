import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        in_0[slice(-8, -4, None), slice(0, -8, None)] = 3
        tmp_0 = in_0
        tmp_0 = None
        return ()