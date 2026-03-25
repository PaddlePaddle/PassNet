import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = torch.nn.functional.interpolate(in_1, None, 2.0, 'nearest', None, recompute_scale_factor = None);  in_1 = None
        tmp_1 = torch.cat([tmp_0, in_0], 1);  tmp_0 = in_0 = None
        return (tmp_1,)
        