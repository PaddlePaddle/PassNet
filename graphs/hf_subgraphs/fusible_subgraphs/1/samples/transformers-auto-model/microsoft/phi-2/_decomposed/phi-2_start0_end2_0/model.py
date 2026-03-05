import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.arange(0, 32, device=device(type='cuda', index=0))
        tmp_1 = tmp_0.unsqueeze(0)
        return (tmp_0, tmp_1)