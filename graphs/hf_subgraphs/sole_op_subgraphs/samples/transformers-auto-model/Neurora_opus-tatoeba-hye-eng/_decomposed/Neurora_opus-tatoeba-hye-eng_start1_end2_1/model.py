import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.full((48, 49), fill_value=-65504.0, dtype=torch.float16, device=device(type='cuda', index=0))
        return (tmp_0,)