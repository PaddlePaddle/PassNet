import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_0 = torch.nn.functional.interpolate(in_0, size = (32, 24), mode = 'nearest');  in_0 = None
        tmp_1 = in_1 * tmp_0;  in_1 = tmp_0 = None
        return (tmp_1,)
        