import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.tensor(-3.4028234663852886e+38, device=device(type='cuda', index=0))
        return (tmp_0,)