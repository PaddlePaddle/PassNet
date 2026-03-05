import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.view(11, 384, 32, 54)
        return (tmp_0,)