import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = torch.conv2d(in_2, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_0 = None
        tmp_2 = in_1 + tmp_1
        tmp_1 = None
        tmp_3 = torch.nn.functional.interpolate(tmp_2, size=[64, 64], mode='bilinear', align_corners=False)
        tmp_2 = None
        return (tmp_3,)