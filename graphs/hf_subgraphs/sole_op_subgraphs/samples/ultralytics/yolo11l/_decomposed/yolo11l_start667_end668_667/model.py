import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.full((400, 1), in_0, dtype=torch.float32, device=device(type='cuda', index=0))
        return (tmp_0,)