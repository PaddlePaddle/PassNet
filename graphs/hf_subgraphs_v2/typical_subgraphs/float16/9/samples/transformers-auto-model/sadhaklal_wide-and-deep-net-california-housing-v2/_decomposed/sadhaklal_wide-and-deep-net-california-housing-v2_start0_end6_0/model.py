import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor):
        linear = torch.nn.functional.linear(in_0, w_1, w_0);  in_0 = w_1 = w_0 = None
        tmp_9 = torch.relu(linear);  linear = None
        linear_1 = torch.nn.functional.linear(tmp_9, w_3, w_2);  tmp_9 = w_3 = w_2 = None
        tmp_11 = torch.relu(linear_1);  linear_1 = None
        tmp_12 = torch.cat([in_1, tmp_11], axis = 1);  in_1 = tmp_11 = None
        linear_2 = torch.nn.functional.linear(tmp_12, w_5, w_4);  tmp_12 = w_5 = w_4 = None
        return (linear_2,)
        