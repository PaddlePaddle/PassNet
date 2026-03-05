import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.tensor(256, dtype=torch.float32, device=device(type='cuda', index=0))
        return (tmp_0,)