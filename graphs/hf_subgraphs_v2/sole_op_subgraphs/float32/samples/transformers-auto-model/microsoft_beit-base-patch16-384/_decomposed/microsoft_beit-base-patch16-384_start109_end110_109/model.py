import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        in_0 *= 47
        tmp_0 = in_0
        return (tmp_0,)