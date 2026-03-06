import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.interpolate(in_0, size=(14, 14), mode='bicubic', antialias=True)
        return (tmp_0,)