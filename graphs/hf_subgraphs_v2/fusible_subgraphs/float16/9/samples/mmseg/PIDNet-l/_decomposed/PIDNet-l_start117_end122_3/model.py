import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        tmp_0 = torch.nn.functional.interpolate(in_2, size = (64, 64), mode = 'bilinear', align_corners = False);  in_2 = None
        tmp_1 = in_0 * tmp_0;  tmp_0 = None
        tmp_2 = 1 - in_0;  in_0 = None
        tmp_3 = tmp_2 * in_1;  tmp_2 = in_1 = None
        tmp_4 = tmp_1 + tmp_3;  tmp_1 = tmp_3 = None
        return (tmp_4,)
        