import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0
        tmp_1 = torch.ones_like(tmp_0, device=device(type='cuda', index=0))
        tmp_0 = None
        return (tmp_1,)