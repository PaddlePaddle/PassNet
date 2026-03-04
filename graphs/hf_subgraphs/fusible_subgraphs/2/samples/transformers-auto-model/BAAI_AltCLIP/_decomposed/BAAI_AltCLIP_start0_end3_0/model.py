import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = tmp_1[slice(None, None, None), slice(None, 7, None)]
        tmp_1 = None
        tmp_3 = tmp_2.expand(2, 7)
        tmp_2 = None
        tmp_4 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        return (tmp_3, tmp_4)