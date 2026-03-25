import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = torch.conv2d(tmp_0, tmp_1, None, (2, 2), (1, 1), (1, 1), 1)
        tmp_0 = tmp_1 = None
        return (tmp_2,)