import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0):
        tmp_0 = w_0[slice(None, None, None), slice(None, 22, None)]
        tmp_1 = tmp_0.expand(1, 22)
        tmp_0 = None
        tmp_2 = in_0[slice(None, None, None), None, None, slice(None, None, None)]
        return (tmp_1, tmp_2)