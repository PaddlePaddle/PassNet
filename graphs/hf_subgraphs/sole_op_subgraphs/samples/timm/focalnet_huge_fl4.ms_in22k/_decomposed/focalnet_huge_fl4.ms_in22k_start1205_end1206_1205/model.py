import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.functional.split(in_0, [2816, 2816, 5], 1)
        return (tmp_0,)