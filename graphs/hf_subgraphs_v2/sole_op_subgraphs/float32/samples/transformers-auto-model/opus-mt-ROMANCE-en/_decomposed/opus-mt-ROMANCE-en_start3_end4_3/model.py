import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.arange(0, 31, dtype=torch.int64, device=device(type='cpu'))
        return (tmp_0,)