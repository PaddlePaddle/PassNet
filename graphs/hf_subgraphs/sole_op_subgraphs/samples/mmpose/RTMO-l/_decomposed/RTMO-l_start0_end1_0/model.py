import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0
        tmp_1 = tmp_0[Ellipsis, slice(None, None, 2), slice(None, None, 2)]
        tmp_0 = None
        return (tmp_1,)