import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.eq(in_0, in_1)
        return (tmp_0,)