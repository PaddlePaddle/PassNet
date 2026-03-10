import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.sum(in_0, dim=2, keepdim=True)
        return (tmp_0,)