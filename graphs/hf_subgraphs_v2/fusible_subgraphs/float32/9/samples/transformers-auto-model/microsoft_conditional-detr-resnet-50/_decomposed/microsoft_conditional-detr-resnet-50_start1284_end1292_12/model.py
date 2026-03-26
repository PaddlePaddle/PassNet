import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = torch.cat([in_0, in_1], dim=3)
        tmp_1 = tmp_0.view(1, 625, 512)
        tmp_0 = None
        tmp_2 = in_2 * 0.125
        tmp_3 = tmp_1.view(1, -1, 8, 64)
        tmp_1 = None
        tmp_4 = tmp_3.transpose(1, 2)
        tmp_3 = None
        tmp_5 = tmp_4.contiguous()
        tmp_4 = None
        tmp_6 = in_3.view(1, -1, 8, 32)
        tmp_7 = tmp_6.transpose(1, 2)
        tmp_6 = None
        return (tmp_5, tmp_2, tmp_7)