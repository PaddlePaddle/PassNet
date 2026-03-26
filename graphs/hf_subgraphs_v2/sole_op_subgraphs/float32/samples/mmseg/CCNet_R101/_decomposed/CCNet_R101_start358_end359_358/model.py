import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.diag(in_0, 0)
        return (tmp_0,)