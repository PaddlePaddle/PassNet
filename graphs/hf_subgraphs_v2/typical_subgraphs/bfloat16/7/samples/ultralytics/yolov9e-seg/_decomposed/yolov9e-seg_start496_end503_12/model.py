import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        tmp_0 = torch.nn.functional.silu(in_0, inplace = True);  in_0 = None
        tmp_1 = torch.cat((in_4, tmp_0), 1);  in_4 = tmp_0 = None
        tmp_2 = torch.nn.functional.interpolate(in_1, size = (80, 80), mode = 'nearest');  in_1 = None
        tmp_3 = torch.nn.functional.interpolate(in_2, size = (80, 80), mode = 'nearest');  in_2 = None
        tmp_4 = torch.nn.functional.interpolate(in_3, size = (80, 80), mode = 'nearest');  in_3 = None
        tmp_5 = torch.stack([tmp_2, tmp_3, tmp_4, tmp_1]);  tmp_2 = tmp_3 = tmp_4 = tmp_1 = None
        tmp_6 = torch.sum(tmp_5, dim = 0);  tmp_5 = None
        return (tmp_6,)
        