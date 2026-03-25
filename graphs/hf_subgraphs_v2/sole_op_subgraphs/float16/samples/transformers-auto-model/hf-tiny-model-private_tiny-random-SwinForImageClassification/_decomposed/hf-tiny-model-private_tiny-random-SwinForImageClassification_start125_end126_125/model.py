import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        in_0[(slice(None, None, None), slice(-2, -1, None), slice(-1, None, None), slice(None, None, None))] = 5;  setitem = in_0;  in_0 = setitem = None
        return ()
        