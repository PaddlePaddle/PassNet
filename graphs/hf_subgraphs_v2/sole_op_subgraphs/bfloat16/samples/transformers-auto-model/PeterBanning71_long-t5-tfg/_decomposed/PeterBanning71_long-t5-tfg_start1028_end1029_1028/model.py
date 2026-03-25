import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.cumsum(in_0, axis = 1);  in_0 = None
        return (tmp_0,)
        