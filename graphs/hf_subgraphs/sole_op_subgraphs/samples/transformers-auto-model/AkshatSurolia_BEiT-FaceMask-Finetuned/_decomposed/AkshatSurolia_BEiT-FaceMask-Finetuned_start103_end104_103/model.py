import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        in_0[slice(1, None, None), slice(1, None, None)] = in_1
        tmp_0 = in_0
        tmp_0 = None
        return ()