import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0 != 0
        tmp_1 = in_0.masked_fill(tmp_0, -1000.0)
        tmp_0 = None
        tmp_2 = in_0 == 0
        return (tmp_2, tmp_1)