import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.interpolate(in_0, size = (128, 128), mode = 'bilinear', align_corners = False);  in_0 = None
        return (tmp_0,)
        