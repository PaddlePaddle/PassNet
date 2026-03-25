import torch
from torch import inf

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.__eq__(inf)
        return (tmp_0,)