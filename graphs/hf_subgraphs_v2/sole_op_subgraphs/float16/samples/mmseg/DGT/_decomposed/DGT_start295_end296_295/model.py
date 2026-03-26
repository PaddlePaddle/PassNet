import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.interpolate(in_0, None, 2, 'bilinear', True);  in_0 = None
        return (tmp_0,)
        