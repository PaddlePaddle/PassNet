import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.zeros((1, 14, 14, 1), dtype=torch.float32, device=device(type='cuda', index=0))
        return (tmp_0,)