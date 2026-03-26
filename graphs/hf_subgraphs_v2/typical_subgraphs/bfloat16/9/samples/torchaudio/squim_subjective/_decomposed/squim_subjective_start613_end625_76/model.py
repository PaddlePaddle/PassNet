import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_4 = torch.cat((in_0, in_1), dim = 2);  in_0 = in_1 = None
        linear = torch.nn.functional.linear(tmp_4, w_1, w_0);  w_1 = w_0 = None
        tmp_6 = linear.transpose(2, 1);  linear = None
        tmp_7 = torch.nn.functional.softmax(tmp_6, dim = 2);  tmp_6 = None
        matmul = torch.matmul(tmp_7, tmp_4);  tmp_7 = tmp_4 = None
        tmp_9 = matmul.squeeze(1);  matmul = None
        linear_1 = torch.nn.functional.linear(tmp_9, w_3, w_2);  tmp_9 = w_3 = w_2 = None
        tmp_11 = torch.nn.functional.softmax(linear_1, dim = 1);  linear_1 = None
        tmp_12 = torch.linspace(0, 4, steps = 5, device = device(type='cuda', index=0))
        tmp_13 = tmp_11 * tmp_12;  tmp_11 = tmp_12 = None
        tmp_14 = tmp_13.sum(dim = 1);  tmp_13 = None
        tmp_15 = 5 - tmp_14;  tmp_14 = None
        return (tmp_15,)
        