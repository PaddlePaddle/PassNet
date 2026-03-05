import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.zeros_like(in_0, device=device(type='cuda', index=0))
        return (tmp_0,)