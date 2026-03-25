import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        tmp_2 = in_2[(slice(None, None, None), 0, slice(None, None, None))];  in_2 = None
        linear = torch.nn.functional.linear(tmp_2, in_0, None);  tmp_2 = in_0 = None
        tmp_4 = in_3.norm(p = 2, dim = -1, keepdim = True)
        tmp_5 = in_3 / tmp_4;  in_3 = tmp_4 = None
        tmp_6 = linear.norm(p = 2, dim = -1, keepdim = True)
        tmp_7 = linear / tmp_6;  linear = tmp_6 = None
        tmp_8 = in_1.exp();  in_1 = None
        tmp_9 = tmp_5.t()
        matmul = torch.matmul(tmp_7, tmp_9);  tmp_9 = None
        tmp_11 = matmul * tmp_8;  matmul = tmp_8 = None
        tmp_12 = tmp_11.t()
        return (tmp_5, tmp_7, tmp_11, tmp_12)
        