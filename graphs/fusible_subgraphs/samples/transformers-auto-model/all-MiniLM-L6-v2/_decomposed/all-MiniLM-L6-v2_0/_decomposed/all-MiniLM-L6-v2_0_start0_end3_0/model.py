import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1):
        tmp_0 = w_1[slice(None, None, None), slice(None, 512, None)]
        tmp_1 = tmp_0.expand(1, 512)
        tmp_0 = None
        tmp_2 = w_0[slice(None, None, None), slice(0, 512, None)]
        return (tmp_1, tmp_2)