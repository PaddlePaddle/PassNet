import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        linear = torch.nn.functional.linear(in_1, w_7, w_6);  in_1 = w_7 = w_6 = None
        tmp_16 = in_2 + linear;  in_2 = linear = None
        tmp_17 = torch.nn.functional.layer_norm(tmp_16, (1152,), w_1, w_0, 1e-06);  w_1 = w_0 = None
        linear_1 = torch.nn.functional.linear(tmp_17, w_3, w_2);  tmp_17 = w_3 = w_2 = None
        tmp_19 = torch.nn.functional.gelu(linear_1, approximate = 'tanh');  linear_1 = None
        linear_2 = torch.nn.functional.linear(tmp_19, w_5, w_4);  tmp_19 = w_5 = w_4 = None
        tmp_21 = tmp_16 + linear_2;  tmp_16 = linear_2 = None
        tmp_22 = torch.nn.functional.layer_norm(tmp_21, (1152,), w_9, w_8, 1e-06);  tmp_21 = w_9 = w_8 = None
        tmp_23 = tmp_22[(slice(None, None, None), -1, slice(None, None, None))]
        linear_3 = torch.nn.functional.linear(tmp_23, w_11, w_10);  tmp_23 = w_11 = w_10 = None
        tmp_25 = in_0.norm(p = 2, dim = -1, keepdim = True)
        tmp_26 = in_0 / tmp_25;  in_0 = tmp_25 = None
        tmp_27 = linear_3.norm(p = 2, dim = -1, keepdim = True)
        tmp_28 = linear_3 / tmp_27;  tmp_27 = None
        tmp_29 = tmp_26.t()
        tmp_30 = tmp_29.to(device(type='cuda'));  tmp_29 = None
        matmul = torch.matmul(tmp_28, tmp_30);  tmp_30 = None
        tmp_32 = w_13.to(device(type='cuda'));  w_13 = None
        tmp_33 = w_12.to(device(type='cuda'));  w_12 = None
        tmp_34 = tmp_32.exp();  tmp_32 = None
        tmp_35 = matmul * tmp_34;  matmul = tmp_34 = None
        tmp_36 = tmp_35 + tmp_33;  tmp_35 = tmp_33 = None
        tmp_37 = tmp_36.t()
        return (tmp_22, linear_3, tmp_26, tmp_28, tmp_36, tmp_37)
        