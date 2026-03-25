import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, in_0 : torch.Tensor):
        linear = torch.nn.functional.linear(in_0, w_1, w_0);  w_1 = w_0 = None
        tmp_5 = linear.view((1, 11, -1, 64));  linear = None
        tmp_6 = tmp_5.transpose(1, 2);  tmp_5 = None
        linear_1 = torch.nn.functional.linear(in_0, w_3, w_2);  in_0 = w_3 = w_2 = None
        tmp_8 = linear_1.view((1, 11, -1, 64));  linear_1 = None
        tmp_9 = tmp_8.transpose(1, 2);  tmp_8 = None
        return (tmp_6, tmp_9)
        