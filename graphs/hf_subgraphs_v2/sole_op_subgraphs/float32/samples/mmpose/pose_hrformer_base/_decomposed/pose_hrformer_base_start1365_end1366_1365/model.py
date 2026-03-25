import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.interpolate(in_0, None, 4.0, 'bilinear', False, recompute_scale_factor=None)
        return (tmp_0,)