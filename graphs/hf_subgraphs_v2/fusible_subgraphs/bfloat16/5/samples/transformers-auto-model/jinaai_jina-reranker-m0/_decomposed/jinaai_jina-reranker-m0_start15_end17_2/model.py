import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_1 = torch.arange(0, 512, device = device(type='cuda', index=0))
        tmp_2 = in_0.to(device = device(type='cuda', index=0), dtype = torch.bool);  in_0 = None
        return (tmp_1, tmp_2)
        