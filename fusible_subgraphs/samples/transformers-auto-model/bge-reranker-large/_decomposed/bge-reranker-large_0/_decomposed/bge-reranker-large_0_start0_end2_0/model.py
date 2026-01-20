import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0):
        tmp_0 = w_0[slice(None, None, None), slice(None, 10, None)]
        tmp_1 = tmp_0.expand(1, 10)
        tmp_0 = None
        return (tmp_1,)