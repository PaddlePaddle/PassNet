import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        in_0[slice(0, -8, None), slice(-8, -4, None)] = 1
        tmp_0 = in_0
        tmp_0 = None
        return ()