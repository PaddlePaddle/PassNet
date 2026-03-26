import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = in_0[(slice(None, None, None), slice(212, None, None))];  in_0 = None
        return (tmp_0,)
        