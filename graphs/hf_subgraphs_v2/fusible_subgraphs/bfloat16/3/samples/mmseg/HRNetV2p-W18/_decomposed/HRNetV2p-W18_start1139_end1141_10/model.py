import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_0 = torch.nn.functional.interpolate(in_3, (128, 128), None, 'bilinear', False);  in_3 = None
        tmp_1 = torch.cat([in_0, in_1, in_2, tmp_0], dim = 1);  in_0 = in_1 = in_2 = tmp_0 = None
        return (tmp_1,)
        