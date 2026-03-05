import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.linspace(0.5, 24.5, 25, dtype=torch.float32, device=device(type='cuda', index=0))
        return (tmp_0,)