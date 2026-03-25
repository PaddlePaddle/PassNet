import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.nn.functional.interpolate(in_0, (128, 128), None, 'bilinear', False)
        tmp_1 = in_1 + tmp_0
        tmp_0 = None
        return (tmp_1,)