import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = in_0
        tmp_3 = torch.conv2d(tmp_2, tmp_1, tmp_0, (2, 2), (1, 1), (1, 1), 1)
        tmp_2 = tmp_1 = tmp_0 = None
        return (tmp_3,)