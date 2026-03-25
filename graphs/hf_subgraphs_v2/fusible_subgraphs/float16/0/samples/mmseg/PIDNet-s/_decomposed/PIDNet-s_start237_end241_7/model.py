import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5):
        conv2d = torch.conv2d(in_5, in_0, None, (1, 1), (0, 0), (1, 1), 1);  in_5 = in_0 = None
        tmp_2 = torch.nn.functional.interpolate(conv2d, size = (8, 8), mode = 'bilinear', align_corners = False);  conv2d = None
        tmp_3 = tmp_2 + in_4;  tmp_2 = in_4 = None
        tmp_4 = torch.cat([in_1, in_2, in_3, tmp_3], dim = 1);  in_1 = in_2 = in_3 = tmp_3 = None
        return (tmp_4,)
        