import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.view(1, 2, -1, 16, 12)
        return (tmp_0,)