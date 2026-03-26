import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        conv2d = torch.conv2d(in_3, in_1, in_0, (2, 2), (1, 1), (1, 1), 1);  in_3 = in_1 = in_0 = None
        tmp_3 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        tmp_4 = in_2 + tmp_3;  in_2 = tmp_3 = None
        tmp_5 = torch.nn.functional.interpolate(tmp_4, size = (24, 24), mode = 'bilinear', align_corners = False);  tmp_4 = None
        return (tmp_5,)
        