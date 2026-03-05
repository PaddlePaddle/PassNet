import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.interpolate(in_0, None, 0.5, 'bilinear', None)
        return (tmp_0,)