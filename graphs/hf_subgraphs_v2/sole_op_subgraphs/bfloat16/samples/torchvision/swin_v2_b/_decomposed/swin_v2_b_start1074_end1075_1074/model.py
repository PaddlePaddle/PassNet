import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        in_0[(slice(-4, None, None), slice(-8, -4, None))] = 7;  setitem = in_0;  in_0 = setitem = None
        return ()
        