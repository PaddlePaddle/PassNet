import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        conv2d = torch.conv2d(in_2, in_1, in_0, (16, 16), (0, 0), (1, 1), 1);  in_2 = in_1 = in_0 = None
        tmp_4 = conv2d.flatten(2);  conv2d = None
        tmp_5 = tmp_4.transpose(1, 2);  tmp_4 = None
        return (tmp_5,)
        