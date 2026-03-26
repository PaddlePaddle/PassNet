import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        tmp_1 = torch.nn.functional.interpolate(in_1, size = (16, 16), mode = 'bilinear', align_corners = False);  in_1 = None
        tmp_2 = torch.cat([in_0, in_2, in_3, in_4, tmp_1], dim = 1);  in_0 = in_2 = in_3 = in_4 = tmp_1 = None
        return (tmp_2,)
        