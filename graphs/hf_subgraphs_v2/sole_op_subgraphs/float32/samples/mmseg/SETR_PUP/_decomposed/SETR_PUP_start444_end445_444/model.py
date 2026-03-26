import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.nn.functional.interpolate(in_1, [in_0, in_0], None, 'bilinear', False)
        return (tmp_0,)