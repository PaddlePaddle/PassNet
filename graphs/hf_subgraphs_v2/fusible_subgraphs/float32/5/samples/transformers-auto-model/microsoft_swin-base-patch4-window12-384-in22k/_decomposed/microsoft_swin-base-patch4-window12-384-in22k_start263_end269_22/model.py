import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0 + in_1
        tmp_1 = tmp_0.view(1, 48, 48, 256)
        tmp_0 = None
        tmp_2 = tmp_1[slice(None, None, None), slice(0, None, 2), slice(0, None, 2), slice(None, None, None)]
        tmp_3 = tmp_1[slice(None, None, None), slice(1, None, 2), slice(0, None, 2), slice(None, None, None)]
        tmp_4 = tmp_1[slice(None, None, None), slice(0, None, 2), slice(1, None, 2), slice(None, None, None)]
        tmp_5 = tmp_1[slice(None, None, None), slice(1, None, 2), slice(1, None, 2), slice(None, None, None)]
        tmp_1 = None
        return (tmp_2, tmp_3, tmp_4, tmp_5)