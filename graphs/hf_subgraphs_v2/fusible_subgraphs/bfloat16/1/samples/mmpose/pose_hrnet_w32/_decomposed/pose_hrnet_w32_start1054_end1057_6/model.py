import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_0 = torch.nn.functional.interpolate(in_0, None, 8.0, 'nearest', None, recompute_scale_factor = None);  in_0 = None
        in_1 += tmp_0;  in_2 = in_1;  in_1 = tmp_0 = None
        tmp_2 = torch.nn.functional.relu(in_2, inplace = True);  in_2 = None
        return (tmp_2,)
        