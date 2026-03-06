import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0 / 6.928203230275509
        return (tmp_0,)