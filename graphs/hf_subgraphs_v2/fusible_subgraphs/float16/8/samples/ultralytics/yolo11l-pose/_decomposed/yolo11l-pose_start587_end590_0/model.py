import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  in_2 = in_1 = in_0 = None
        tmp_3 = conv2d.view(64, 51, -1);  conv2d = None
        tmp_4 = torch.cat([in_3, in_4, tmp_3], -1);  in_3 = in_4 = tmp_3 = None
        return (tmp_4,)
        