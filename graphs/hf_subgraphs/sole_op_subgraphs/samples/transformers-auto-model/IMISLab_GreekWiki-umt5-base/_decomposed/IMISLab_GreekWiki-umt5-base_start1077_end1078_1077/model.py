import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.zeros((1, 12, 12, 12), device=device(type='cuda', index=0), dtype=torch.float32)
        return (tmp_0,)