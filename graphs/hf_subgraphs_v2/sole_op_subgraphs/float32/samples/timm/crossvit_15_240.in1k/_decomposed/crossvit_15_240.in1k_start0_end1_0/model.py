import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0
        tmp_1 = torch.nn.functional.interpolate(tmp_0, size=(240, 240), mode='bicubic', align_corners=False)
        tmp_0 = None
        return (tmp_1,)