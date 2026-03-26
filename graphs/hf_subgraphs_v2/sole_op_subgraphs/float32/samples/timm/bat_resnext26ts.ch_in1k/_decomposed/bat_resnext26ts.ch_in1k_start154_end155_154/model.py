import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.functional.split(in_0, 1, dim=2)
        return (tmp_0,)