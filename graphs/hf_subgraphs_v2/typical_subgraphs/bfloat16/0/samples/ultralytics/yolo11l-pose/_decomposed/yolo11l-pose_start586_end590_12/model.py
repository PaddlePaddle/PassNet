import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        tmp_2 = torch.nn.functional.silu(in_2, inplace = True);  in_2 = None
        conv2d = torch.conv2d(tmp_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_2 = in_1 = in_0 = None
        tmp_4 = conv2d.view(1, 51, -1);  conv2d = None
        tmp_5 = torch.cat([in_3, in_4, tmp_4], -1);  in_3 = in_4 = tmp_4 = None
        return (tmp_5,)
        