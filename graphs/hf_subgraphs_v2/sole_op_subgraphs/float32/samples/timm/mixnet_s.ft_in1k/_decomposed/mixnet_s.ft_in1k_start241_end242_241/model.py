import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.functional.split(in_0, [160, 160, 160], 1)
        return (tmp_0,)