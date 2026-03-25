import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_0 = torch.nn.functional.interpolate(in_0, size = (64, 48), mode = 'nearest');  in_0 = None
        tmp_1 = in_2 * tmp_0;  in_2 = tmp_0 = None
        tmp_2 = torch.nn.functional.interpolate(in_1, size = (32, 24), mode = 'nearest');  in_1 = None
        tmp_3 = in_3 * tmp_2;  in_3 = tmp_2 = None
        return (tmp_1, tmp_3)
        