import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_1 * in_0
        tmp_1 = in_1[Ellipsis, slice(None, 32, None)]
        tmp_2 = in_1[Ellipsis, slice(32, None, None)]
        return (tmp_0, tmp_1, tmp_2)