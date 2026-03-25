import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.interpolate(in_0, (160, 160), None, 'bilinear', False)
        return (tmp_0,)