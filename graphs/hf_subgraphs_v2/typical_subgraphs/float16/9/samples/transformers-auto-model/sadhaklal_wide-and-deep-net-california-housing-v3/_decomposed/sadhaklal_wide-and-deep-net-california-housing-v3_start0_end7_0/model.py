import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor):
        linear = torch.nn.functional.linear(in_0, w_3, w_2);  in_0 = w_3 = w_2 = None
        tmp_11 = torch.relu(linear);  linear = None
        linear_1 = torch.nn.functional.linear(tmp_11, w_5, w_4);  tmp_11 = w_5 = w_4 = None
        tmp_13 = torch.relu(linear_1);  linear_1 = None
        tmp_14 = torch.cat([in_1, tmp_13], dim = 1);  in_1 = None
        linear_2 = torch.nn.functional.linear(tmp_14, w_7, w_6);  tmp_14 = w_7 = w_6 = None
        linear_3 = torch.nn.functional.linear(tmp_13, w_1, w_0);  tmp_13 = w_1 = w_0 = None
        return (linear_2, linear_3)
        