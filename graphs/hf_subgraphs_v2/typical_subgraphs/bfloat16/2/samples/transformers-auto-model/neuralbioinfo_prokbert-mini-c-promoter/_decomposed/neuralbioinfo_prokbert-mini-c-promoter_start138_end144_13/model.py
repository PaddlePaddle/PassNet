import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        linear = torch.nn.functional.linear(in_4, in_1, in_0);  in_1 = in_0 = None
        tmp_5 = linear.view(16, -1, 6, 64);  linear = None
        tmp_6 = tmp_5.transpose(1, 2);  tmp_5 = None
        linear_1 = torch.nn.functional.linear(in_4, in_3, in_2);  in_4 = in_3 = in_2 = None
        tmp_8 = linear_1.view(16, -1, 6, 64);  linear_1 = None
        tmp_9 = tmp_8.transpose(1, 2);  tmp_8 = None
        return (tmp_6, tmp_9)
        