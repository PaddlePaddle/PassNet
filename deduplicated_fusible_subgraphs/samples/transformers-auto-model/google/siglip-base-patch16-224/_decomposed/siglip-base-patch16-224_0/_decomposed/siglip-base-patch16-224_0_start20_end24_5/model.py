import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, w_0):
        tmp_0 = in_1 + in_2
        tmp_1 = tmp_0[slice(None, None, None), 0]
        tmp_0 = None
        tmp_2 = in_0.view(-1, 64)
        tmp_3 = w_0[slice(None, None, None), slice(None, 64, None)]
        return (tmp_1, tmp_2, tmp_3)