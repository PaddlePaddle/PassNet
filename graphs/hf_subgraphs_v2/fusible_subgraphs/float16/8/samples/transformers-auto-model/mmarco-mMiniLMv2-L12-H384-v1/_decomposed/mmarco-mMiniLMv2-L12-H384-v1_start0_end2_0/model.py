import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_1 = in_0[(slice(None, None, None), slice(None, 64, None))];  in_0 = None
        tmp_2 = tmp_1.expand(128, 64);  tmp_1 = None
        return (tmp_2,)
        