import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, in_0, in_1):
        tmp_6 = 0.5 * in_1
        tmp_7 = torch.pow(in_1, 3.0)
        tmp_8 = 0.044715 * tmp_7;  tmp_7 = None
        tmp_9 = in_1 + tmp_8;  in_1 = tmp_8 = None
        tmp_10 = 0.7978845608028654 * tmp_9;  tmp_9 = None
        tmp_11 = torch.tanh(tmp_10);  tmp_10 = None
        tmp_12 = 1.0 + tmp_11;  tmp_11 = None
        tmp_13 = tmp_6 * tmp_12;  tmp_6 = tmp_12 = None
        linear = torch.nn.functional.linear(tmp_13, w_3, w_2);  tmp_13 = w_3 = w_2 = None
        tmp_15 = torch.nn.functional.dropout(linear, 0.1, False, False);  linear = None
        tmp_16 = tmp_15 + in_0;  tmp_15 = in_0 = None
        tmp_17 = torch.nn.functional.layer_norm(tmp_16, (32,), w_1, w_0, 1e-12);  tmp_16 = w_1 = w_0 = None
        tmp_18 = tmp_17[(slice(None, None, None), 0, slice(None, None, None))]
        linear_1 = torch.nn.functional.linear(tmp_18, w_5, w_4);  tmp_18 = w_5 = w_4 = None
        tmp_20 = torch.tanh(linear_1);  linear_1 = None
        return (tmp_17, tmp_20)
        