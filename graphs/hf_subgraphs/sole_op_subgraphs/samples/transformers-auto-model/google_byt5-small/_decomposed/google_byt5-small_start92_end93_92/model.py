import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.zeros((1, 6, 50, 50), device=device(type='cuda', index=0), dtype=torch.float32)
        return (tmp_0,)