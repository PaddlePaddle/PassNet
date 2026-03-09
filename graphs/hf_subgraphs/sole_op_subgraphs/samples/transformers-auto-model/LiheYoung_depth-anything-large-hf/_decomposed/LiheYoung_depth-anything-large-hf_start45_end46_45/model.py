import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.interpolate(in_0, size=(74, 74), mode='bilinear', align_corners=True)
        return (tmp_0,)