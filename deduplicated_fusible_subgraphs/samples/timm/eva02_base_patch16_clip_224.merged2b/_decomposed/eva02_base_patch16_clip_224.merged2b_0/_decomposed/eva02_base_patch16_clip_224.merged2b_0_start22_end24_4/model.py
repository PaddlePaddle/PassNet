import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0 * in_1
        tmp_1 = in_0[Ellipsis, slice(1, None, 2)]
        return (tmp_0, tmp_1)