import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_0, in_1, in_2):
        linear = torch.nn.functional.linear(in_1, w_1, w_0);  in_1 = w_1 = w_0 = None
        tmp_11 = torch.nn.functional.silu(in_2, inplace = False);  in_2 = None
        tmp_12 = tmp_11 * linear;  tmp_11 = linear = None
        tmp_13 = torch.nn.functional.dropout(tmp_12, 0.0, False, False);  tmp_12 = None
        tmp_14 = torch.nn.functional.layer_norm(tmp_13, (2048,), w_5, w_4, 1e-06);  tmp_13 = w_5 = w_4 = None
        linear_1 = torch.nn.functional.linear(tmp_14, w_3, w_2);  tmp_14 = w_3 = w_2 = None
        tmp_16 = torch.nn.functional.dropout(linear_1, 0.0, False, False);  linear_1 = None
        tmp_17 = in_0 + tmp_16;  in_0 = tmp_16 = None
        tmp_18 = torch.nn.functional.layer_norm(tmp_17, (768,), w_9, w_8, 1e-06);  tmp_17 = w_9 = w_8 = None
        tmp_19 = tmp_18[(slice(None, None, None), 0)];  tmp_18 = None
        tmp_20 = torch.nn.functional.dropout(tmp_19, 0.0, False, False);  tmp_19 = None
        linear_2 = torch.nn.functional.linear(tmp_20, w_7, w_6);  tmp_20 = w_7 = w_6 = None
        return (linear_2,)
        