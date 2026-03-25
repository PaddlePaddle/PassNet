import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1);  in_1 = in_0 = None
        tmp_2 = in_2 + conv2d;  in_2 = conv2d = None
        split = torch.functional.split(tmp_2, [192, 576], dim = 1)
        tmp_4 = split[0]
        tmp_5 = split[1];  split = None
        return (tmp_4, tmp_5, tmp_2)
        