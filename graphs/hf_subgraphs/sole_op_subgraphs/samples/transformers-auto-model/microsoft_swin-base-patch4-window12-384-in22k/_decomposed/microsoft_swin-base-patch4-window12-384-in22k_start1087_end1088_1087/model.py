import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        in_0[slice(None, None, None), slice(-6, None, None), slice(0, -12, None), slice(None, None, None)] = 6
        tmp_0 = in_0
        tmp_0 = None
        return ()