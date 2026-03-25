import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9):
        tmp_8 = torch.nn.functional.gelu(in_9);  in_9 = None
        linear = torch.nn.functional.linear(tmp_8, in_3, in_2);  tmp_8 = in_3 = in_2 = None
        tmp_10 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_11 = tmp_10 + in_8;  tmp_10 = in_8 = None
        tmp_12 = torch.nn.functional.layer_norm(tmp_11, (192,), in_5, in_4, 1e-12);  tmp_11 = in_5 = in_4 = None
        tmp_13 = tmp_12[(slice(None, None, None), 0, slice(None, None, None))]
        linear_1 = torch.nn.functional.linear(tmp_13, in_1, in_0);  tmp_13 = in_1 = in_0 = None
        tmp_15 = tmp_12[(slice(None, None, None), 1, slice(None, None, None))];  tmp_12 = None
        linear_2 = torch.nn.functional.linear(tmp_15, in_7, in_6);  tmp_15 = in_7 = in_6 = None
        tmp_17 = linear_1 + linear_2
        tmp_18 = tmp_17 / 2;  tmp_17 = None
        return (linear_1, linear_2, tmp_18)
        