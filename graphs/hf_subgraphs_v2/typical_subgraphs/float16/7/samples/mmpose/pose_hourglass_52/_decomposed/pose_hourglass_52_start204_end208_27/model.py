import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        in_2 += in_0;  in_3 = in_2;  in_2 = in_0 = None
        tmp_1 = torch.nn.functional.relu(in_3, inplace = True);  in_3 = None
        tmp_2 = torch.nn.functional.interpolate(tmp_1, None, 2.0, 'nearest', None, recompute_scale_factor = None);  tmp_1 = None
        tmp_3 = in_1 + tmp_2;  in_1 = tmp_2 = None
        return (tmp_3,)
        