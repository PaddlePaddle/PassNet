import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = in_0[(slice(None, None, None), 1, Ellipsis)];  in_0 = None
        return (tmp_0,)
        