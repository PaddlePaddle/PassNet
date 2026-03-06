import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0):
        tmp_0 = w_0
        tmp_1 = tmp_0[slice(None, 2209, None)]
        tmp_0 = None
        return (tmp_1,)