import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.full((33, 34), fill_value=-3.4028234663852886e+38, dtype=torch.float32, device=device(type='cuda', index=0))
        tmp_1 = torch.triu(tmp_0, diagonal=1)
        tmp_0 = None
        return (tmp_1,)