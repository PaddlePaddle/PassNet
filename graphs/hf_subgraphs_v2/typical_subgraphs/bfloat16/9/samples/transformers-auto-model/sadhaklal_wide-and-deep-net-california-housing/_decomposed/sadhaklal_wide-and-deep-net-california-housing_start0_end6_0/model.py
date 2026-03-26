import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, in_0 : torch.Tensor):
        linear = torch.nn.functional.linear(in_0, w_1, w_0);  w_1 = w_0 = None
        tmp_8 = torch.relu(linear);  linear = None
        linear_1 = torch.nn.functional.linear(tmp_8, w_3, w_2);  tmp_8 = w_3 = w_2 = None
        tmp_10 = torch.relu(linear_1);  linear_1 = None
        tmp_11 = torch.cat([in_0, tmp_10], axis = 1);  in_0 = tmp_10 = None
        linear_2 = torch.nn.functional.linear(tmp_11, w_5, w_4);  tmp_11 = w_5 = w_4 = None
        return (linear_2,)
        