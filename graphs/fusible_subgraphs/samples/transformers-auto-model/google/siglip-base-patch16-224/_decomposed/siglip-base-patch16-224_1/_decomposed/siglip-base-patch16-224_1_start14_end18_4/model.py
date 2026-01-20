import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0.norm(p=2, dim=-1, keepdim=True)
        tmp_1 = in_0 / tmp_0
        tmp_0 = None
        tmp_2 = in_1.t()
        tmp_3 = tmp_2.to(device(type='cuda'))
        tmp_2 = None
        return (tmp_1, tmp_3)