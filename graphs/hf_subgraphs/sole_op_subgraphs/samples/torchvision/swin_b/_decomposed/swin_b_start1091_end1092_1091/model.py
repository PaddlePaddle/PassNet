import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.view(2, 7, 2, 7)
        return (tmp_0,)