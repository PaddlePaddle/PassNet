import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.arange(13, device=device(type='cuda', index=0), dtype=torch.int64)
        return (tmp_0,)