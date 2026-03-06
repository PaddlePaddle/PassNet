import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.zeros((1, 16, 16), device=device(type='cuda', index=0), dtype=torch.bool)
        return (tmp_0,)