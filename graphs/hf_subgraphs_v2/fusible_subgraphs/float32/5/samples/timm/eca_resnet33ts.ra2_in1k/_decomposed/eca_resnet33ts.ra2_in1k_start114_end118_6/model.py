import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_1.sigmoid()
        tmp_1 = tmp_0.view(1, -1, 1, 1)
        tmp_0 = None
        tmp_2 = tmp_1.expand_as(in_0)
        tmp_1 = None
        tmp_3 = in_0 * tmp_2
        tmp_2 = None
        return (tmp_3,)