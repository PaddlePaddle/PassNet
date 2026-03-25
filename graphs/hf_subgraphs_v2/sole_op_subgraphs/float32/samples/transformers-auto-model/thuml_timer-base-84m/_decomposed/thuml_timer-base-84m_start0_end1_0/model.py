import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0
        tmp_1 = tmp_0.unfold(dimension=-1, size=96, step=96)
        tmp_0 = None
        return (tmp_1,)