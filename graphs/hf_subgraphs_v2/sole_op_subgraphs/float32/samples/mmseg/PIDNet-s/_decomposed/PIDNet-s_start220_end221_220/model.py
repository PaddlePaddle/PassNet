import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.interpolate(in_0, size=(8, 8), mode='bilinear', align_corners=False)
        return (tmp_0,)