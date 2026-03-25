import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.clamp(in_0, min = 1e-09);  in_0 = None
        return (tmp_0,)
        