import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        in_0[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 1)] = in_1;  setitem = in_0;  in_0 = in_1 = setitem = None
        return ()
        