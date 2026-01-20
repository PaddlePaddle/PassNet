import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1):
        tmp_0 = w_1[slice(None, None, None), slice(None, 36, None)]
        tmp_1 = tmp_0.expand(2, 36)
        tmp_0 = None
        tmp_2 = w_0[slice(None, None, None), slice(0, 36, None)]
        return (tmp_1, tmp_2)