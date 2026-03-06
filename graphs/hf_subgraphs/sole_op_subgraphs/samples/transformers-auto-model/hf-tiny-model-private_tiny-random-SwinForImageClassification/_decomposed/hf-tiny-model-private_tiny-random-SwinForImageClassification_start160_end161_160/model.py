import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.view(1, 16, 2, 4, 4)
        return (tmp_0,)