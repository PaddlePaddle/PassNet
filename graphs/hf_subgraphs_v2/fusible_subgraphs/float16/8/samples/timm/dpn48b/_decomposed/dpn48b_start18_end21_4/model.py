import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        conv2d = torch.conv2d(in_2, in_0, None, (1, 1), (0, 0), (1, 1), 1);  in_2 = in_0 = None
        tmp_2 = in_3 + in_1;  in_3 = in_1 = None
        tmp_3 = torch.cat([in_4, conv2d], dim = 1);  in_4 = conv2d = None
        return (tmp_3, tmp_2)
        