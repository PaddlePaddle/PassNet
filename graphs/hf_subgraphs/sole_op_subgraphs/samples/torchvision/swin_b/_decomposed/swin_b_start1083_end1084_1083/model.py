import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        in_0[slice(0, -7, None), slice(-7, -3, None)] = 1
        tmp_0 = in_0
        tmp_0 = None
        return ()