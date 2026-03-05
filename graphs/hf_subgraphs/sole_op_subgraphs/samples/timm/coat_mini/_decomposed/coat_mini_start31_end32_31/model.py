import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.functional.split(in_0, [38, 57, 57], dim=1)
        return (tmp_0,)