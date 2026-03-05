import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.interpolate(in_0, size=(512, 512), mode='bilinear', align_corners=False)
        tmp_0 = None
        return ()