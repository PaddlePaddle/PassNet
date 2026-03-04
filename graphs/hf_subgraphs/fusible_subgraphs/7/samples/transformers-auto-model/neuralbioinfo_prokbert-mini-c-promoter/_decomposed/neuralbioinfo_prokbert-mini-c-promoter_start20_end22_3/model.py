import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.arange(128, dtype=torch.int64, device=device(type='cuda', index=0))
        tmp_1 = tmp_0.view(-1, 1)
        tmp_0 = None
        return (tmp_1,)