import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, in_0, in_1, in_2):
        tmp_1 = torch.nn.functional.silu(in_0, inplace = True);  in_0 = None
        tmp_2 = w_0.view(-1, 768, 1, 1);  w_0 = None
        tmp_3 = tmp_2 * tmp_1;  tmp_2 = tmp_1 = None
        tmp_4 = in_2 + tmp_3;  in_2 = tmp_3 = None
        tmp_5 = torch.nn.functional.interpolate(tmp_4, None, 2.0, 'nearest', None, recompute_scale_factor = None)
        tmp_6 = torch.cat([tmp_5, in_1], 1);  tmp_5 = in_1 = None
        return (tmp_4, tmp_6)
        