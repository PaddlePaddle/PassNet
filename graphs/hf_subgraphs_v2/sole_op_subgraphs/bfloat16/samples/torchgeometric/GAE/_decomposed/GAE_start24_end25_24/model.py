import torch

from torch import inf

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = in_0.__eq__(inf);  in_0 = None
        return (tmp_0,)
        