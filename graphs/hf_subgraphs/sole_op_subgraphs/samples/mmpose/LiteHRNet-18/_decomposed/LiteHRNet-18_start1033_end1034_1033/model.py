import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.interpolate(in_0, size=(64, 48), mode='nearest')
        return (tmp_0,)