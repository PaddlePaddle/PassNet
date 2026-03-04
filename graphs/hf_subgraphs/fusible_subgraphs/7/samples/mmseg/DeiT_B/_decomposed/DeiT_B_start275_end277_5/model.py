import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.nn.functional.interpolate(in_0, (32, 32), None, 'bilinear', False)
        tmp_1 = torch.nn.functional.interpolate(in_1, (32, 32), None, 'bilinear', False)
        return (tmp_0, tmp_1)