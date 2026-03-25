import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.le(in_0, 5);  in_0 = None
        return (tmp_0,)
        