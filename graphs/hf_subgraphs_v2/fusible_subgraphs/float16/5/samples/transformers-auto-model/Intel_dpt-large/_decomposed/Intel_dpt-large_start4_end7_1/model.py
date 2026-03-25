import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.interpolate(in_0, size = (24, 24), mode = 'bilinear');  in_0 = None
        tmp_1 = tmp_0.permute(0, 2, 3, 1);  tmp_0 = None
        tmp_2 = tmp_1.reshape(1, 576, -1);  tmp_1 = None
        return (tmp_2,)
        