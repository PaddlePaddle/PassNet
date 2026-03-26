import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = in_0[(slice(None, None, None), slice(3752, 6247, None))];  in_0 = None
        return (tmp_0,)
        