import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0[Ellipsis, slice(0, None, 2), slice(1, None, 2), slice(None, None, None)]
        return (tmp_0,)