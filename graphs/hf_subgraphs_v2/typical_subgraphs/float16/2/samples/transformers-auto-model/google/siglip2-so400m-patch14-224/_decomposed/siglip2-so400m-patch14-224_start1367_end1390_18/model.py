import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor):
        linear = torch.nn.functional.linear(in_14, in_7, in_6);  in_14 = in_7 = in_6 = None
        tmp_15 = in_15 + linear;  in_15 = linear = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (1152,), in_1, in_0, 1e-06);  in_1 = in_0 = None
        linear_1 = torch.nn.functional.linear(tmp_16, in_3, in_2);  tmp_16 = in_3 = in_2 = None
        tmp_18 = torch.nn.functional.gelu(linear_1, approximate = 'tanh');  linear_1 = None
        linear_2 = torch.nn.functional.linear(tmp_18, in_5, in_4);  tmp_18 = in_5 = in_4 = None
        tmp_20 = tmp_15 + linear_2;  tmp_15 = linear_2 = None
        tmp_21 = torch.nn.functional.layer_norm(tmp_20, (1152,), in_9, in_8, 1e-06);  tmp_20 = in_9 = in_8 = None
        tmp_22 = tmp_21[(slice(None, None, None), -1, slice(None, None, None))]
        linear_3 = torch.nn.functional.linear(tmp_22, in_11, in_10);  tmp_22 = in_11 = in_10 = None
        tmp_24 = in_16.norm(p = 2, dim = -1, keepdim = True)
        tmp_25 = in_16 / tmp_24;  in_16 = tmp_24 = None
        tmp_26 = linear_3.norm(p = 2, dim = -1, keepdim = True)
        tmp_27 = linear_3 / tmp_26;  tmp_26 = None
        tmp_28 = tmp_25.t()
        tmp_29 = tmp_28.to(device(type='cuda'));  tmp_28 = None
        matmul = torch.matmul(tmp_27, tmp_29);  tmp_29 = None
        tmp_31 = in_13.to(device(type='cuda'));  in_13 = None
        tmp_32 = in_12.to(device(type='cuda'));  in_12 = None
        tmp_33 = tmp_31.exp();  tmp_31 = None
        tmp_34 = matmul * tmp_33;  matmul = tmp_33 = None
        tmp_35 = tmp_34 + tmp_32;  tmp_34 = tmp_32 = None
        tmp_36 = tmp_35.t()
        return (tmp_21, linear_3, tmp_25, tmp_27, tmp_35, tmp_36)
        