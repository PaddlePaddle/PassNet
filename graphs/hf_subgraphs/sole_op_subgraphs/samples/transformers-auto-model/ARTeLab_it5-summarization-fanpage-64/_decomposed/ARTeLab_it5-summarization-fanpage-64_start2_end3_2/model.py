import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.triu(in_0, diagonal=1)
        return (tmp_0,)