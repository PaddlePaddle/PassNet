import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_0 = torch.sigmoid(in_0)
        tmp_1 = in_0 * tmp_0;  in_0 = tmp_0 = None
        tmp_2 = torch.nn.functional.interpolate(tmp_1, None, 2.0, 'nearest', None, recompute_scale_factor = None)
        tmp_3 = torch.cat([tmp_2, in_1], 1);  tmp_2 = in_1 = None
        return (tmp_3, tmp_1)
        