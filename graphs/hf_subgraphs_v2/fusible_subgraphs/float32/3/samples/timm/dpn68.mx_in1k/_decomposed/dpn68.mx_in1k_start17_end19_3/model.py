import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = torch.conv2d(in_1, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_0 = None
        tmp_2 = tmp_1[slice(None, None, None), slice(None, 64, None), slice(None, None, None), slice(None, None, None)]
        return (tmp_2, tmp_1)