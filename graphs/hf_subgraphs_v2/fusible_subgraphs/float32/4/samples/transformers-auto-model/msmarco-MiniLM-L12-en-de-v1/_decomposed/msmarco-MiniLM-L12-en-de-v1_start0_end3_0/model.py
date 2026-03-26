import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = tmp_1[slice(None, None, None), slice(None, 256, None)]
        tmp_1 = None
        tmp_3 = tmp_2.expand(8, 256)
        tmp_2 = None
        tmp_4 = tmp_0[slice(None, None, None), slice(0, 256, None)]
        tmp_0 = None
        return (tmp_3, tmp_4)