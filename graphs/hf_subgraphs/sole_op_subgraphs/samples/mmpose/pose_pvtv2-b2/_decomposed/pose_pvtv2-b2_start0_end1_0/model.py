import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = w_1
        tmp_3 = torch.conv2d(tmp_0, tmp_2, tmp_1, (4, 4), (3, 3), (1, 1), 1)
        tmp_0 = tmp_2 = tmp_1 = None
        return (tmp_3,)