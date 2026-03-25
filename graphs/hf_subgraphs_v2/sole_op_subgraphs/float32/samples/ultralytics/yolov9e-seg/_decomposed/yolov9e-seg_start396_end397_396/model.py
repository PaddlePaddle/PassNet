import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.sum(in_0, dim=0)
        return (tmp_0,)