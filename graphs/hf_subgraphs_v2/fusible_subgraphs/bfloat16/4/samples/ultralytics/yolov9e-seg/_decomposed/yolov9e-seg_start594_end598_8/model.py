import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_0 = torch.cat((in_2, in_3), 1);  in_2 = in_3 = None
        tmp_1 = torch.nn.functional.interpolate(in_0, size = (40, 40), mode = 'nearest');  in_0 = None
        tmp_2 = torch.nn.functional.interpolate(in_1, size = (40, 40), mode = 'nearest');  in_1 = None
        tmp_3 = torch.stack([tmp_1, tmp_2, tmp_0]);  tmp_1 = tmp_2 = tmp_0 = None
        return (tmp_3,)
        