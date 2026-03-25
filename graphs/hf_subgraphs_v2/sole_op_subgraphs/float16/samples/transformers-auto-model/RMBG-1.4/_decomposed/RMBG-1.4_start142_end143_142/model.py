import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.interpolate(in_0, size = (40, 40), mode = 'bilinear');  in_0 = None
        return (tmp_0,)
        