import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        linear = torch.nn.functional.linear(in_3, in_2, in_1);  in_2 = in_1 = None
        tmp_4 = linear.view(1, -1, 12, 64);  linear = None
        tmp_5 = tmp_4.transpose(1, 2);  tmp_4 = None
        linear_1 = torch.nn.functional.linear(in_3, in_0, None);  in_3 = in_0 = None
        tmp_7 = linear_1.view(1, -1, 12, 64);  linear_1 = None
        tmp_8 = tmp_7.transpose(1, 2);  tmp_7 = None
        return (tmp_8, tmp_5)
        