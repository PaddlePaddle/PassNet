import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0
        tmp_1 = tmp_0[slice(None, None, None), slice(None, 1, None)]
        tmp_2 = tmp_0[0, slice(1, None, None)]
        tmp_0 = None
        tmp_3 = tmp_2.reshape(1, 24, 24, -1)
        tmp_2 = None
        tmp_4 = tmp_3.permute(0, 3, 1, 2)
        tmp_3 = None
        return (tmp_4, tmp_1)