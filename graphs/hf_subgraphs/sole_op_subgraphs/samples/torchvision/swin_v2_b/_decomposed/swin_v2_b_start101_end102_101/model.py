import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.view(7, 8, 7, 8)
        return (tmp_0,)