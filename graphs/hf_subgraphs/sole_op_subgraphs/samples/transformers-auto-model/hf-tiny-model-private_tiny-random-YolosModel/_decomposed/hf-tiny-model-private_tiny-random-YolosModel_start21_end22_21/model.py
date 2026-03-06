import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0):
        tmp_0 = w_0
        tmp_1 = tmp_0[slice(None, None, None), slice(None, None, None), slice(1, -10, None), slice(None, None, None)]
        tmp_0 = None
        return (tmp_1,)