import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, in_1 : torch.Tensor):
        tmp_10 = torch.nn.functional.dropout(in_1, 0.1, False, False)
        linear = torch.nn.functional.linear(tmp_10, w_1, w_0);  tmp_10 = w_1 = w_0 = None
        tmp_12 = 0.5 * linear
        tmp_13 = torch.pow(linear, 3.0)
        tmp_14 = 0.044715 * tmp_13;  tmp_13 = None
        tmp_15 = linear + tmp_14;  linear = tmp_14 = None
        tmp_16 = 0.7978845608028654 * tmp_15;  tmp_15 = None
        tmp_17 = torch.tanh(tmp_16);  tmp_16 = None
        tmp_18 = 1.0 + tmp_17;  tmp_17 = None
        tmp_19 = tmp_12 * tmp_18;  tmp_12 = tmp_18 = None
        linear_1 = torch.nn.functional.linear(tmp_19, w_5, w_4);  tmp_19 = w_5 = w_4 = None
        tmp_21 = torch.nn.functional.dropout(linear_1, 0.1, False, False);  linear_1 = None
        tmp_22 = tmp_21 + in_1;  tmp_21 = in_1 = None
        tmp_23 = torch.nn.functional.layer_norm(tmp_22, (1024,), w_3, w_2, 1e-12);  tmp_22 = w_3 = w_2 = None
        linear_2 = torch.nn.functional.linear(tmp_23, w_7, w_6);  tmp_23 = w_7 = w_6 = None
        tmp_25 = in_0 * 1000000.0;  in_0 = None
        tmp_26 = linear_2 - tmp_25;  linear_2 = tmp_25 = None
        split = tmp_26.split(1, dim = -1);  tmp_26 = None
        tmp_28 = split[0]
        tmp_29 = split[1];  split = None
        tmp_30 = tmp_28.squeeze(-1);  tmp_28 = None
        tmp_31 = tmp_30.contiguous();  tmp_30 = None
        tmp_32 = tmp_29.squeeze(-1);  tmp_29 = None
        tmp_33 = tmp_32.contiguous();  tmp_32 = None
        return (tmp_31, tmp_33)
        