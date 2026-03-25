import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5):
        conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  in_3 = in_1 = in_0 = None
        tmp_4 = conv2d + in_5;  conv2d = in_5 = None
        tmp_5 = in_2.unsqueeze(-1);  in_2 = None
        tmp_6 = tmp_5.unsqueeze(-1);  tmp_5 = None
        tmp_7 = tmp_6 * tmp_4;  tmp_6 = tmp_4 = None
        tmp_8 = in_4 + tmp_7;  in_4 = tmp_7 = None
        return (tmp_8,)
        