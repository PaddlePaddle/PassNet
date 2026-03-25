import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.conv2d(in_0, in_1, None, (2, 2), (1, 1), (1, 1), 1)
        return (tmp_0,)