import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.interpolate(in_0, (32, 32), None, 'nearest', None)
        return (tmp_0,)