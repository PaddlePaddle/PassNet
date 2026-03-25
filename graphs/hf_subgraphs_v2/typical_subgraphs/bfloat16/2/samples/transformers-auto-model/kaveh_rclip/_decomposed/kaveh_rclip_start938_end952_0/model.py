import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor):
        tmp_5 = in_5[(slice(None, None, None), 0)];  in_5 = None
        linear = torch.nn.functional.linear(tmp_5, in_1, in_0);  tmp_5 = in_1 = in_0 = None
        tmp_7 = torch.tanh(linear);  linear = None
        linear_1 = torch.nn.functional.linear(in_6, in_3, None);  in_6 = in_3 = None
        linear_2 = torch.nn.functional.linear(tmp_7, in_2, None);  in_2 = None
        tmp_10 = linear_1.norm(dim = -1, keepdim = True)
        tmp_11 = linear_1 / tmp_10;  linear_1 = tmp_10 = None
        tmp_12 = linear_2.norm(dim = -1, keepdim = True)
        tmp_13 = linear_2 / tmp_12;  linear_2 = tmp_12 = None
        tmp_14 = in_4.exp();  in_4 = None
        tmp_15 = tmp_11.t()
        matmul = torch.matmul(tmp_13, tmp_15);  tmp_15 = None
        tmp_17 = matmul * tmp_14;  matmul = tmp_14 = None
        tmp_18 = tmp_17.T
        return (tmp_7, tmp_11, tmp_13, tmp_17, tmp_18)
        