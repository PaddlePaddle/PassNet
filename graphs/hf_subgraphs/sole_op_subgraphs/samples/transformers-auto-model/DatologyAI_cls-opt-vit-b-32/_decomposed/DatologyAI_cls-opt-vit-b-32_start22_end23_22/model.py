import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0
        tmp_1 = torch.cumsum(tmp_0, dim=1)
        tmp_0 = None
        return (tmp_1,)