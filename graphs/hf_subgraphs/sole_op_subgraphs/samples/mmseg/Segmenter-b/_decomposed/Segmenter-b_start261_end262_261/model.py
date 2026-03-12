import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.normalize(in_0, dim=2, p=2)
        return (tmp_0,)