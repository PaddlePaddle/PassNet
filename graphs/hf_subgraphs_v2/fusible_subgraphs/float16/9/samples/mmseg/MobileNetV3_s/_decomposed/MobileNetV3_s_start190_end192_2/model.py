import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        conv2d = torch.conv2d(in_2, in_0, None, (1, 1), (0, 0), (1, 1), 1);  in_2 = in_0 = None
        tmp_2 = torch.cat([in_1, conv2d], 1);  in_1 = conv2d = None
        return (tmp_2,)
        