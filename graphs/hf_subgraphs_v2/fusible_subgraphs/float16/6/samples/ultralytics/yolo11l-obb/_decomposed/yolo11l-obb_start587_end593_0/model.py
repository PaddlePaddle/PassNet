import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  in_2 = in_1 = in_0 = None
        tmp_3 = conv2d.view(24, 1, -1);  conv2d = None
        tmp_4 = torch.cat([in_3, in_4, tmp_3], 2);  in_3 = in_4 = tmp_3 = None
        tmp_5 = tmp_4.sigmoid();  tmp_4 = None
        tmp_6 = tmp_5 - 0.25;  tmp_5 = None
        tmp_7 = tmp_6 * 3.141592653589793;  tmp_6 = None
        return (tmp_7,)
        