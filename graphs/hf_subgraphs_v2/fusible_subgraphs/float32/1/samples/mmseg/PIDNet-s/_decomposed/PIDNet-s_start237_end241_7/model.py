import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = torch.conv2d(in_5, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_0 = None
        tmp_2 = torch.nn.functional.interpolate(tmp_1, size=(8, 8), mode='bilinear', align_corners=False)
        tmp_1 = None
        tmp_3 = tmp_2 + in_4
        tmp_2 = None
        tmp_4 = torch.cat([in_1, in_2, in_3, tmp_3], dim=1)
        tmp_3 = None
        return (tmp_4,)