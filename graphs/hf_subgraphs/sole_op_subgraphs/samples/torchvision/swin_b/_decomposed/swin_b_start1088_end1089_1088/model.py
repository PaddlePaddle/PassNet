import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        in_0[slice(-3, None, None), slice(0, -7, None)] = 6
        tmp_0 = in_0
        tmp_0 = None
        return ()