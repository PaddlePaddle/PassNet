import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        conv2d = torch.conv2d(in_2, in_0, None, (1, 1), (0, 0), (1, 1), 1);  in_2 = in_0 = None
        tmp_2 = torch.nn.functional.interpolate(conv2d, size = (8, 8), mode = 'bilinear', align_corners = False);  conv2d = None
        tmp_3 = tmp_2 + in_1;  tmp_2 = in_1 = None
        return (tmp_3,)
        