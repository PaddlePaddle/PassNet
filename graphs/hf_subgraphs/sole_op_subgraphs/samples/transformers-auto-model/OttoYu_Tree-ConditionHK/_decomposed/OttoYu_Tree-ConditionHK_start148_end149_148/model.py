import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.view((16, 49, -1, 32))
        return (tmp_0,)