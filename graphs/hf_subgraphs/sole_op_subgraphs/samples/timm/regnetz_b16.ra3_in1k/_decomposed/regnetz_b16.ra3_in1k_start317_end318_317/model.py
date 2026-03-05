import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, in_0):
        tmp_0 = w_0
        tmp_1 = torch.conv2d(in_0, tmp_0, None, (1, 1), (1, 1), (1, 1), 54)
        tmp_0 = None
        return (tmp_1,)