import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        in_1[(slice(None, None, None), slice(0, None, 3))] = in_0;  setitem = in_1;  in_1 = in_0 = setitem = None
        return ()
        