import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        in_0[slice(0, None, None), 0] = 2210
        tmp_0 = in_0
        tmp_0 = None
        return ()