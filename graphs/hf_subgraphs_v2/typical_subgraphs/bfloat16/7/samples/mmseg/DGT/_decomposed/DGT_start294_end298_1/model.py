import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        tmp_2 = in_4 + in_2;  in_4 = in_2 = None
        tmp_3 = torch.nn.functional.interpolate(tmp_2, None, 2, 'bilinear', True);  tmp_2 = None
        conv2d = torch.conv2d(tmp_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_3 = in_1 = in_0 = None
        tmp_5 = in_3.clone();  in_3 = None
        return (tmp_5, conv2d)
        