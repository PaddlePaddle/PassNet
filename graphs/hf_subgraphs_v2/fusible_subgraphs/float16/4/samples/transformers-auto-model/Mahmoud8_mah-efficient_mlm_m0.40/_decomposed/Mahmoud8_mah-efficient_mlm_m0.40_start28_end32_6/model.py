import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        linear = torch.nn.functional.linear(in_2, in_1, in_0);  in_2 = in_1 = in_0 = None
        tmp_3 = linear.view(8, -1, 16, 64);  linear = None
        tmp_4 = tmp_3.transpose(1, 2);  tmp_3 = None
        tmp_5 = in_3.transpose(-1, -2);  in_3 = None
        return (tmp_5, tmp_4)
        