import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.silu(in_2, inplace=True)
        tmp_3 = torch.conv2d(tmp_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_2 = tmp_1 = tmp_0 = None
        tmp_4 = tmp_3.view(12, 32, -1)
        tmp_3 = None
        tmp_5 = torch.cat([in_3, in_4, tmp_4], 2)
        tmp_4 = None
        return (tmp_5,)