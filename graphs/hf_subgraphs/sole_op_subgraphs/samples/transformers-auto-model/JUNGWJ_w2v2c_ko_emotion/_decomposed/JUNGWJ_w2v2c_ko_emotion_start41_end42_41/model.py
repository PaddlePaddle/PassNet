import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, w_0):
        tmp_0 = w_0
        tmp_1 = tmp_0.to(dtype=torch.float32, device=device(type='cuda', index=0))
        tmp_0 = None
        return (tmp_1,)