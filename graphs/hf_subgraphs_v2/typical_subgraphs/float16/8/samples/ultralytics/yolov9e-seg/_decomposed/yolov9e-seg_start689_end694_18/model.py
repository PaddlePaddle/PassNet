import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        tmp_0 = torch.nn.functional.silu(in_0, inplace = True);  in_0 = None
        tmp_1 = torch.cat((in_2, tmp_0), 1);  in_2 = tmp_0 = None
        tmp_2 = torch.nn.functional.interpolate(in_1, size = (20, 20), mode = 'nearest');  in_1 = None
        tmp_3 = torch.stack([tmp_2, tmp_1]);  tmp_2 = tmp_1 = None
        tmp_4 = torch.sum(tmp_3, dim = 0);  tmp_3 = None
        return (tmp_4,)
        