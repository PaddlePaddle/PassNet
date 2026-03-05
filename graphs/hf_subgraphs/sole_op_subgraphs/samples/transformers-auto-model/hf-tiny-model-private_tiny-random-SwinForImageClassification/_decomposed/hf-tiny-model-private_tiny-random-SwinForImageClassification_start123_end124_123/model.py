import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        in_0[slice(None, None, None), slice(-2, -1, None), slice(0, -2, None), slice(None, None, None)] = 3
        tmp_0 = in_0
        tmp_0 = None
        return ()