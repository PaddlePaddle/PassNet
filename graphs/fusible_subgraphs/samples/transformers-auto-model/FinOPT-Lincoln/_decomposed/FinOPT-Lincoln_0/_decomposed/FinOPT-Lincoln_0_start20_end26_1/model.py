import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.cumsum(in_0, dim=1)
        tmp_1 = tmp_0 * in_0
        tmp_0 = None
        tmp_2 = tmp_1 - 1
        tmp_1 = None
        tmp_3 = tmp_2.long()
        tmp_2 = None
        tmp_4 = tmp_3[slice(None, None, None), slice(0, None, None)]
        tmp_3 = None
        tmp_5 = tmp_4 + 2
        tmp_4 = None
        return (tmp_5,)