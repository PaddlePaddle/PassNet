import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12):
        linear = torch.nn.functional.linear(in_11, in_1, in_0);  in_11 = in_1 = in_0 = None
        tmp_11 = torch.nn.functional.silu(in_12, inplace = False);  in_12 = None
        tmp_12 = tmp_11 * linear;  tmp_11 = linear = None
        tmp_13 = torch.nn.functional.dropout(tmp_12, 0.0, False, False);  tmp_12 = None
        tmp_14 = torch.nn.functional.layer_norm(tmp_13, (2048,), in_5, in_4, 1e-06);  tmp_13 = in_5 = in_4 = None
        linear_1 = torch.nn.functional.linear(tmp_14, in_3, in_2);  tmp_14 = in_3 = in_2 = None
        tmp_16 = torch.nn.functional.dropout(linear_1, 0.0, False, False);  linear_1 = None
        tmp_17 = in_10 + tmp_16;  in_10 = tmp_16 = None
        tmp_18 = torch.nn.functional.layer_norm(tmp_17, (768,), in_9, in_8, 1e-06);  tmp_17 = in_9 = in_8 = None
        tmp_19 = tmp_18[(slice(None, None, None), 0)];  tmp_18 = None
        tmp_20 = torch.nn.functional.dropout(tmp_19, 0.0, False, False);  tmp_19 = None
        linear_2 = torch.nn.functional.linear(tmp_20, in_7, in_6);  tmp_20 = in_7 = in_6 = None
        return (linear_2,)
        