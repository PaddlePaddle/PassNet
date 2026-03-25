import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.roll(in_0, shifts = (-4, -4), dims = (1, 2));  in_0 = None
        return (tmp_0,)
        