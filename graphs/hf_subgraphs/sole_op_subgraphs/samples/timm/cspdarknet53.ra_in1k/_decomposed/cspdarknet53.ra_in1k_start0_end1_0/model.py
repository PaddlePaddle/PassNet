import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, in_0):
        tmp_0 = w_0
        tmp_1 = in_0
        tmp_2 = torch.conv2d(tmp_1, tmp_0, None, (1, 1), (1, 1), (1, 1), 1)
        tmp_1 = tmp_0 = None
        return (tmp_2,)