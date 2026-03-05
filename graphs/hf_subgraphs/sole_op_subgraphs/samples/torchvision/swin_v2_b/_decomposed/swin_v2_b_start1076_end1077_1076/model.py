import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.view(2, 8, 2, 8)
        return (tmp_0,)