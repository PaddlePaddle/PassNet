import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  in_2 = in_1 = in_0 = None
        split = conv2d.split([64, 128, 256, 512], dim = 1);  conv2d = None
        tmp_4 = split[0]
        tmp_5 = split[1]
        tmp_6 = split[2]
        tmp_7 = split[3];  split = None
        return (tmp_4, tmp_5, tmp_6, tmp_7)
        