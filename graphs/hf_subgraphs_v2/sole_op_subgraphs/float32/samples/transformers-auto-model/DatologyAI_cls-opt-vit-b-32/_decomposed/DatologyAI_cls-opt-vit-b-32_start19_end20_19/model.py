import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.all(in_0, dim=-1, keepdim=True)
        return (tmp_0,)