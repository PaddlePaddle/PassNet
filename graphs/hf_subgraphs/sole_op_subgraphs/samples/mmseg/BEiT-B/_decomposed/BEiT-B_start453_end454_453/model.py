import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = torch.conv_transpose2d(in_0, tmp_1, tmp_0, (2, 2), (0, 0), (0, 0), 1, (1, 1))
        tmp_1 = tmp_0 = None
        return (tmp_2,)