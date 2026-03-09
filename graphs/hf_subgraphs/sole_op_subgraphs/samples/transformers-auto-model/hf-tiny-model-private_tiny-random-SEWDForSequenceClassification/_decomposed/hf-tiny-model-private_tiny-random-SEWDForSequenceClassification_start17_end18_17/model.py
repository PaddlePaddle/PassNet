import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0[Ellipsis, slice(None, 3999, None)]
        return (tmp_0,)