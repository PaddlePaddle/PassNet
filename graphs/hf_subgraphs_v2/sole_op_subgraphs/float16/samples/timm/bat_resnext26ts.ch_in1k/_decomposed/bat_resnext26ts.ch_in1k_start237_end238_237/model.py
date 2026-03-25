import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self):
        tmp_0 = torch.eye(4, 4, dtype = torch.float32, device = device(type='cpu'))
        return (tmp_0,)
        