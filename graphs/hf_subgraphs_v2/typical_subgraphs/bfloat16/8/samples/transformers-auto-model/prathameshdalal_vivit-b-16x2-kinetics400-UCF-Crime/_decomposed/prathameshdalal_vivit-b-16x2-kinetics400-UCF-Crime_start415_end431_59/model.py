import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_6 = 0.5 * in_7
        tmp_7 = in_7 * 0.7978845608
        tmp_8 = 0.044715 * in_7
        tmp_9 = tmp_8 * in_7;  tmp_8 = in_7 = None
        tmp_10 = 1.0 + tmp_9;  tmp_9 = None
        tmp_11 = tmp_7 * tmp_10;  tmp_7 = tmp_10 = None
        tmp_12 = torch.tanh(tmp_11);  tmp_11 = None
        tmp_13 = 1.0 + tmp_12;  tmp_12 = None
        tmp_14 = tmp_6 * tmp_13;  tmp_6 = tmp_13 = None
        tmp_15 = torch.nn.functional.dropout(tmp_14, 0.0, False, False);  tmp_14 = None
        linear = torch.nn.functional.linear(tmp_15, in_3, in_2);  tmp_15 = in_3 = in_2 = None
        tmp_17 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_18 = tmp_17 + in_6;  tmp_17 = in_6 = None
        tmp_19 = torch.nn.functional.layer_norm(tmp_18, (768,), in_5, in_4, 1e-06);  tmp_18 = in_5 = in_4 = None
        tmp_20 = tmp_19[(slice(None, None, None), 0, slice(None, None, None))];  tmp_19 = None
        linear_1 = torch.nn.functional.linear(tmp_20, in_1, in_0);  tmp_20 = in_1 = in_0 = None
        return (linear_1,)
        