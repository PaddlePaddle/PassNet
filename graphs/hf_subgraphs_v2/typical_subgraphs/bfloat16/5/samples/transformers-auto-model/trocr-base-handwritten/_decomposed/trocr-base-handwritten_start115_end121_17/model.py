import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        linear = torch.nn.functional.linear(in_2, in_1, None);  in_1 = None
        tmp_3 = linear.view(1, -1, 12, 64);  linear = None
        tmp_4 = tmp_3.transpose(1, 2);  tmp_3 = None
        linear_1 = torch.nn.functional.linear(in_2, in_0, None);  in_2 = in_0 = None
        tmp_6 = linear_1.view(1, -1, 12, 64);  linear_1 = None
        tmp_7 = tmp_6.transpose(1, 2);  tmp_6 = None
        return (tmp_7, tmp_4)
        