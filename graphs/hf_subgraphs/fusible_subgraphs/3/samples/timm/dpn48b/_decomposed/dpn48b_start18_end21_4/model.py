import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = torch.conv2d(in_2, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_0 = None
        tmp_2 = in_3 + in_1
        tmp_3 = torch.cat([in_4, tmp_1], dim=1)
        tmp_1 = None
        return (tmp_3, tmp_2)