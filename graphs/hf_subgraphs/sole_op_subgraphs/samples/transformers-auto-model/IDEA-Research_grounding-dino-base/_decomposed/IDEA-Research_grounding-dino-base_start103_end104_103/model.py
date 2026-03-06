import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.full((1, 900, 256), -1000000.0, device=device(type='cuda', index=0))
        return (tmp_0,)