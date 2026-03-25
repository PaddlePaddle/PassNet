import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = torch.nn.functional.interpolate(in_1, size=(16, 16), mode='bilinear', align_corners=False)
        tmp_2 = torch.cat([tmp_0, in_2, in_3, in_4, tmp_1], dim=1)
        tmp_0 = tmp_1 = None
        return (tmp_2,)