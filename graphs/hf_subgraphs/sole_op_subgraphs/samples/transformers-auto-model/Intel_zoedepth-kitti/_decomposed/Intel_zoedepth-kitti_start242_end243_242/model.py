import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.softplus(in_0, 1.0, 20.0)
        return (tmp_0,)