import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.arange(0, 1000, device=device(type='cuda'))
        tmp_1 = tmp_0.view(1, -1)
        tmp_0 = None
        tmp_2 = tmp_1.repeat(2, 1)
        tmp_1 = None
        return (tmp_2,)