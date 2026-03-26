import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.pow(in_0, 3.0)
        return (tmp_0,)