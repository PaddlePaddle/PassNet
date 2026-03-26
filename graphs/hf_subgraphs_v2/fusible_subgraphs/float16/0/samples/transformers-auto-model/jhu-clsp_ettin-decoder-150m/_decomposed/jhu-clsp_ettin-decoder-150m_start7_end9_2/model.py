import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self):
        tmp_0 = torch.arange(64, device = device(type='cuda', index=0))
        tmp_0 += 0;  tmp_1 = tmp_0;  tmp_0 = None
        return (tmp_1,)
        