import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_0 = torch.nn.functional.silu(in_0, inplace = True);  in_0 = None
        tmp_1 = torch.cat((in_3, tmp_0), 1);  in_3 = tmp_0 = None
        tmp_2 = torch.nn.functional.interpolate(in_1, size = (40, 40), mode = 'nearest');  in_1 = None
        tmp_3 = torch.nn.functional.interpolate(in_2, size = (40, 40), mode = 'nearest');  in_2 = None
        tmp_4 = torch.stack([tmp_2, tmp_3, tmp_1]);  tmp_2 = tmp_3 = tmp_1 = None
        tmp_5 = torch.sum(tmp_4, dim = 0);  tmp_4 = None
        return (tmp_5,)
        