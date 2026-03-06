import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.interpolate(in_0, (64, 64), mode='bilinear', align_corners=False)
        return (tmp_0,)