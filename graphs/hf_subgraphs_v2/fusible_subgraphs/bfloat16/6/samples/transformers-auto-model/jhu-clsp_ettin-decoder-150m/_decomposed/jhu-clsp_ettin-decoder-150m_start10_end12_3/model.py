import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self):
        tmp_0 = torch.arange(1, device = device(type='cuda', index=0))
        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions = None
        return (tmp_0,)
        