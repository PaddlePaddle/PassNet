import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.interpolate(in_0, None, 4.0, 'nearest', None, recompute_scale_factor = None);  in_0 = None
        return (tmp_0,)
        