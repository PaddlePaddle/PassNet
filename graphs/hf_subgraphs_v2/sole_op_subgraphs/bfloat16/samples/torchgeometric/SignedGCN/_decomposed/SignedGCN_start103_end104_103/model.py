import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = in_0[(Ellipsis, slice(32, None, None))];  in_0 = None
        return (tmp_0,)
        