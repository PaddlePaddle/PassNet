import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0
        tmp_1 = tmp_0[slice(None, None, None), slice(None, 7, None)]
        tmp_0 = None
        tmp_2 = tmp_1.expand(2, 7)
        tmp_1 = None
        return (tmp_2,)