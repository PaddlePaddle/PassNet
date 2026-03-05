import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.zeros((0, 0, 0), dtype=torch.float32, device=device(type='cuda', index=0))
        tmp_0 = None
        return ()