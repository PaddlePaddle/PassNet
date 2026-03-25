import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor, in_17 : torch.Tensor):
        linear = torch.nn.functional.linear(in_16, in_7, in_6);  in_16 = in_7 = in_6 = None
        tmp_17 = in_17 + linear;  in_17 = linear = None
        tmp_18 = torch.nn.functional.layer_norm(tmp_17, (1536,), in_1, in_0, 1e-06);  in_1 = in_0 = None
        to = tmp_18.to(torch.float16);  tmp_18 = None
        linear_1 = torch.nn.functional.linear(to, in_3, in_2);  to = in_3 = in_2 = None
        tmp_20 = torch.nn.functional.gelu(linear_1, approximate = 'tanh');  linear_1 = None
        to_1 = tmp_20.to(torch.float16);  tmp_20 = None
        linear_2 = torch.nn.functional.linear(to_1, in_5, in_4);  to_1 = in_5 = in_4 = None
        tmp_22 = tmp_17 + linear_2;  tmp_17 = linear_2 = None
        tmp_23 = torch.nn.functional.layer_norm(tmp_22, (1536,), in_9, in_8, 1e-06);  in_9 = in_8 = None
        to_2 = tmp_23.to(torch.float16)
        linear_3 = torch.nn.functional.linear(to_2, in_13, in_12);  to_2 = in_13 = in_12 = None
        to_3 = tmp_23.to(torch.float16)
        linear_4 = torch.nn.functional.linear(to_3, in_11, in_10);  to_3 = in_11 = in_10 = None
        to_4 = tmp_23.to(torch.float16);  tmp_23 = None
        linear_5 = torch.nn.functional.linear(to_4, in_15, in_14);  to_4 = in_15 = in_14 = None
        return (tmp_22, linear_4, linear_3, linear_5)
        