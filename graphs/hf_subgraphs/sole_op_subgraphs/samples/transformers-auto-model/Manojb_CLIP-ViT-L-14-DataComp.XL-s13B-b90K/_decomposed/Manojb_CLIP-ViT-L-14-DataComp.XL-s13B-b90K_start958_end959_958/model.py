import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.to(dtype=torch.int32, device=device(type='cuda', index=0))
        return (tmp_0,)