import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self):
        tmp_0 = torch.arange(19, device = device(type='cpu'), dtype = torch.float32)
        return (tmp_0,)
        