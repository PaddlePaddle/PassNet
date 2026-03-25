import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  in_2 = in_1 = in_0 = None
        tmp_3 = torch.stack([conv2d], dim = 0);  conv2d = None
        tmp_4 = tmp_3.sum(dim = 0);  tmp_3 = None
        tmp_5 = torch.cat([tmp_4, in_3], 1);  tmp_4 = in_3 = None
        return (tmp_5,)
        