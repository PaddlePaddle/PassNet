import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.split([36, 36, 72], dim=2)
        return (tmp_0,)