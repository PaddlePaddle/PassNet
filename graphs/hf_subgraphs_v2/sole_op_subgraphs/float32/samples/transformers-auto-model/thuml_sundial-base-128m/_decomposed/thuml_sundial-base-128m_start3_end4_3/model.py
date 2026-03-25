import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.unfold(dimension=-1, size=16, step=16)
        return (tmp_0,)