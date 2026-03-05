import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.view(1, 2, 2, 8, 8, 384)
        return (tmp_0,)