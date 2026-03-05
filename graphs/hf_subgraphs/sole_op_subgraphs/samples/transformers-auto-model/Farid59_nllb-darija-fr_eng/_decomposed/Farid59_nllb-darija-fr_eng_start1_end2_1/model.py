import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.full((16, 17), fill_value=-3.3895313892515355e+38, dtype=torch.bfloat16, device=device(type='cuda', index=0))
        return (tmp_0,)