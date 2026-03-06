import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.interpolate(in_0, size=(48, 48), mode='bilinear', align_corners=False)
        return (tmp_0,)