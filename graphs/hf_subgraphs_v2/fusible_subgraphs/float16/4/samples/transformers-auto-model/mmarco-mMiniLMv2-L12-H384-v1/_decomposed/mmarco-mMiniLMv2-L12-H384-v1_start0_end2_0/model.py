import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_1 = in_0[(slice(None, None, None), slice(None, 256, None))];  in_0 = None
        tmp_2 = tmp_1.expand(8, 256);  tmp_1 = None
        return (tmp_2,)
        