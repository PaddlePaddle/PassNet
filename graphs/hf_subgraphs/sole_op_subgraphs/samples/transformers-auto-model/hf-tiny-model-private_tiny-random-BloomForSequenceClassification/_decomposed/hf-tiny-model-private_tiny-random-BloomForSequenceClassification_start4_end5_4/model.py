import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.arange(1, 5, device=device(type='cuda', index=0), dtype=torch.int32)
        return (tmp_0,)