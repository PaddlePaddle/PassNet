import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = tmp_0.exp()
        tmp_0 = None
        tmp_2 = tmp_1.to(device=device(type='cuda', index=0))
        tmp_1 = None
        tmp_3 = in_1.to(device=device(type='cuda', index=0), dtype=torch.float32)
        tmp_4 = tmp_3.t()
        return (tmp_3, tmp_2, tmp_4)