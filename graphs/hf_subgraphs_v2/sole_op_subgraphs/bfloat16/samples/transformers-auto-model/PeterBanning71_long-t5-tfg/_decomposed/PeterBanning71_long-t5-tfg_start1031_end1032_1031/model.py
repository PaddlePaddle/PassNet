import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.where(in_0, 1.0, -1000.0);  in_0 = None
        return (tmp_0,)
        