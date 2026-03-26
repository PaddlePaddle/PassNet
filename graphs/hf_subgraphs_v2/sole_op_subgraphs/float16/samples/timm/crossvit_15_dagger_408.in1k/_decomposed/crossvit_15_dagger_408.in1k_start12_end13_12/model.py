import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_1 = torch.nn.functional.interpolate(in_0, size = (384, 384), mode = 'bicubic', align_corners = False);  in_0 = None
        return (tmp_1,)
        