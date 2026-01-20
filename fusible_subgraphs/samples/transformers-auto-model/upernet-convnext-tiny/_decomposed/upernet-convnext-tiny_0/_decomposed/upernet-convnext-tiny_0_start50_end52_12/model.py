import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = torch.nn.functional.interpolate(in_1, size=(128, 128), mode='bilinear', align_corners=False)
        tmp_1 = torch.cat([in_0, tmp_0, in_3, in_2], dim=1)
        tmp_0 = None
        return (tmp_1,)