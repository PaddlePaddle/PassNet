import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.pow(in_0, 0.5)
        return (tmp_0,)