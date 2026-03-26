import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.interpolate(in_0, size = (37, 37), mode = 'bilinear', align_corners = True);  in_0 = None
        return (tmp_0,)
        