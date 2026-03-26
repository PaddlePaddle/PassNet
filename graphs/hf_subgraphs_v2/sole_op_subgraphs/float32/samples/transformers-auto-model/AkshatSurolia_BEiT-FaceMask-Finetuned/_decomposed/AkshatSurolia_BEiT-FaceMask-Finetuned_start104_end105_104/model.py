import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        in_0[0, slice(0, None, None)] = 729
        tmp_0 = in_0
        tmp_0 = None
        return ()