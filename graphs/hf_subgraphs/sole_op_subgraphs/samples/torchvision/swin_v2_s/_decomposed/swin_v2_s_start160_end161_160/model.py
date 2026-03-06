import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.view(1, 4, 8, 4, 8, 192)
        return (tmp_0,)