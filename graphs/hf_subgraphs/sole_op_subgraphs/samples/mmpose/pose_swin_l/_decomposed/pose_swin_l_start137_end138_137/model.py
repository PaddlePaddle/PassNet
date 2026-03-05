import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.view(1, 5, 7, 4, 7, 384)
        return (tmp_0,)