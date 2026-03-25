import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = torch.conv2d(in_1, in_0, tmp_0, (2, 2), (0, 0), (1, 1), 2)
        tmp_0 = None
        return (tmp_1,)