import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.ones(1, 131, 131, dtype=torch.bool, device=device(type='cuda', index=0))
        return (tmp_0,)