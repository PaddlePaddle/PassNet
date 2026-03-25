import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0
        tmp_1 = tmp_0[slice(None, None, None), slice(None, 35, None)]
        tmp_0 = None
        tmp_2 = tmp_1.expand(2, 35)
        tmp_1 = None
        return (tmp_2,)