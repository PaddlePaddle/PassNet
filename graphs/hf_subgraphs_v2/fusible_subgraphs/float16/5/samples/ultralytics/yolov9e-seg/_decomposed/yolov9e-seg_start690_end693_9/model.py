import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        tmp_0 = torch.cat((in_1, in_2), 1);  in_1 = in_2 = None
        tmp_1 = torch.nn.functional.interpolate(in_0, size = (20, 20), mode = 'nearest');  in_0 = None
        tmp_2 = torch.stack([tmp_1, tmp_0]);  tmp_1 = tmp_0 = None
        return (tmp_2,)
        