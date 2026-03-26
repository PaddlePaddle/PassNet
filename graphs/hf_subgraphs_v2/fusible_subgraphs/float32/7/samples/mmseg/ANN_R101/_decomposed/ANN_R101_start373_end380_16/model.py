import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_1, 8)
        tmp_1 = tmp_0.view(24, 256, -1)
        tmp_0 = None
        tmp_2 = torch.cat([in_2, in_3, in_4, tmp_1], dim=2)
        tmp_1 = None
        tmp_3 = in_0.reshape(24, 256, -1)
        tmp_4 = tmp_2.reshape(24, 256, -1)
        tmp_2 = None
        tmp_5 = tmp_4.permute(0, 2, 1)
        tmp_4 = None
        tmp_6 = tmp_5.contiguous()
        tmp_5 = None
        return (tmp_3, tmp_6)