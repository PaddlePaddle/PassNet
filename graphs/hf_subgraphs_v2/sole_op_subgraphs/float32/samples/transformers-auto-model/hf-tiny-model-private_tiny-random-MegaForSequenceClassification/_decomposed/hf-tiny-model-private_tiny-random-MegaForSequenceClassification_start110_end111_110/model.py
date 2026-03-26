import torch
from torch import inf

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0.masked_fill(in_1, -inf)
        return (tmp_0,)