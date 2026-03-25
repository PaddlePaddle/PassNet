import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        in_0[(slice(-7, -3, None), slice(0, -7, None))] = 3;  setitem = in_0;  in_0 = setitem = None
        return ()
        