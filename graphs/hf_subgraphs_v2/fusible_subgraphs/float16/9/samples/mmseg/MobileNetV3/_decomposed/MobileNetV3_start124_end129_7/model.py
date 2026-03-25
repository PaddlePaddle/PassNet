import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  in_3 = in_1 = in_0 = None
        tmp_3 = conv2d + 3.0;  conv2d = None
        tmp_4 = tmp_3 / 6.0;  tmp_3 = None
        tmp_5 = tmp_4.clamp_(0.0, 1.0);  tmp_4 = None
        tmp_6 = in_2 * tmp_5;  in_2 = tmp_5 = None
        return (tmp_6,)
        