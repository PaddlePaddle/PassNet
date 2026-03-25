import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_3 = tmp_2.view(12, 51, -1)
        tmp_2 = None
        tmp_4 = torch.cat([in_3, in_4, tmp_3], -1)
        tmp_3 = None
        return (tmp_4,)