import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.as_tensor([(100, 100), (50, 50), (25, 25), (13, 13)], dtype=torch.int64, device=device(type='cuda', index=0))
        return (tmp_0,)