import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = torch.nn.functional.interpolate(in_1, (64, 128), None, 'nearest', None);  in_1 = None
        tmp_1 = in_0 + tmp_0;  in_0 = tmp_0 = None
        return (tmp_1,)
        