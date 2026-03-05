import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.functional.split(in_0, [2048, 2048, 4], 1)
        return (tmp_0,)