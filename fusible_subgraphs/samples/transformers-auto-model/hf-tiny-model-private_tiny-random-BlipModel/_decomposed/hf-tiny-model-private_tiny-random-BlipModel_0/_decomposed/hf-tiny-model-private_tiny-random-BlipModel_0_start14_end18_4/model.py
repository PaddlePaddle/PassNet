import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0):
        tmp_0 = w_0.exp()
        tmp_1 = tmp_0.to(device=device(type='cuda', index=0))
        tmp_0 = None
        tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.float32)
        tmp_3 = tmp_2.t()
        return (tmp_1, tmp_2, tmp_3)