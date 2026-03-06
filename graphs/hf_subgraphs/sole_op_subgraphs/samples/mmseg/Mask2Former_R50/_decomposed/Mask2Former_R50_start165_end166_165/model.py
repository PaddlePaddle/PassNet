import torch
from torch import inf

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_1.masked_fill_(in_0, -inf)
        return (tmp_0,)