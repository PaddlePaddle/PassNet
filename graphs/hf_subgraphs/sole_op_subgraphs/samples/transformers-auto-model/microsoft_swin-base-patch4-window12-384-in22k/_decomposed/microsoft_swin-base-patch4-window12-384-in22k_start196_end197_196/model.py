import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.zeros((1, 48, 48, 1), dtype=torch.float32, device=device(type='cuda', index=0))
        return (tmp_0,)