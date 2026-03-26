import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor):
        linear = torch.nn.functional.linear(in_0, w_7, w_6);  in_0 = w_7 = w_6 = None
        tmp_17 = in_1 + linear;  in_1 = linear = None
        tmp_18 = torch.nn.functional.layer_norm(tmp_17, (1024,), w_1, w_0, 1e-06);  w_1 = w_0 = None
        linear_1 = torch.nn.functional.linear(tmp_18, w_3, w_2);  tmp_18 = w_3 = w_2 = None
        tmp_20 = torch.nn.functional.gelu(linear_1, approximate = 'tanh');  linear_1 = None
        linear_2 = torch.nn.functional.linear(tmp_20, w_5, w_4);  tmp_20 = w_5 = w_4 = None
        tmp_22 = tmp_17 + linear_2;  tmp_17 = linear_2 = None
        tmp_23 = torch.nn.functional.layer_norm(tmp_22, (1024,), w_9, w_8, 1e-06);  w_9 = w_8 = None
        linear_3 = torch.nn.functional.linear(tmp_23, w_13, w_12);  w_13 = w_12 = None
        linear_4 = torch.nn.functional.linear(tmp_23, w_11, w_10);  w_11 = w_10 = None
        linear_5 = torch.nn.functional.linear(tmp_23, w_15, w_14);  tmp_23 = w_15 = w_14 = None
        return (tmp_22, linear_4, linear_3, linear_5)
        