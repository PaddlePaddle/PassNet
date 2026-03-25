import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        in_0 /= 256.0
        tmp_0 = in_0
        return (tmp_0,)