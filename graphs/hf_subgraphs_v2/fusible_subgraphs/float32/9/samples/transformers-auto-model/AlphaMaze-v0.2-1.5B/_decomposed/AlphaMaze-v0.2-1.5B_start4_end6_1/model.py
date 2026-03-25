import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.arange(20, device=device(type='cuda'))
        tmp_0 += 0
        tmp_1 = tmp_0
        tmp_0 = None
        return (tmp_1,)