import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        in_0[slice(-7, -3, None), slice(-3, None, None)] = 5
        tmp_0 = in_0
        tmp_0 = None
        return ()