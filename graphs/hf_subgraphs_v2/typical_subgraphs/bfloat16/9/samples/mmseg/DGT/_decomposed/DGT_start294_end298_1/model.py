import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, in_0, in_1, in_2):
        tmp_2 = in_2 + in_0;  in_2 = in_0 = None
        tmp_3 = torch.nn.functional.interpolate(tmp_2, None, 2, 'bilinear', True);  tmp_2 = None
        conv2d = torch.conv2d(tmp_3, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_3 = w_1 = w_0 = None
        tmp_5 = in_1.clone();  in_1 = None
        return (tmp_5, conv2d)
        