import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, in_0 : torch.Tensor):
        linear = torch.nn.functional.linear(in_0, w_2, w_1);  w_2 = w_1 = None
        tmp_4 = linear.view(4, -1, 4, 16);  linear = None
        tmp_5 = tmp_4.transpose(1, 2);  tmp_4 = None
        linear_1 = torch.nn.functional.linear(in_0, w_0, None);  in_0 = w_0 = None
        tmp_7 = linear_1.view(4, -1, 4, 16);  linear_1 = None
        tmp_8 = tmp_7.transpose(1, 2);  tmp_7 = None
        return (tmp_8, tmp_5)
        