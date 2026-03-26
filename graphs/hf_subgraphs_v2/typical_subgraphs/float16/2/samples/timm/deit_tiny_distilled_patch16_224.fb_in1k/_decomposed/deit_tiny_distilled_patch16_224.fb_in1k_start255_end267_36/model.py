import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9):
        tmp_8 = torch.nn.functional.gelu(in_9, approximate = 'none');  in_9 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False);  tmp_8 = None
        linear = torch.nn.functional.linear(tmp_9, in_1, in_0);  tmp_9 = in_1 = in_0 = None
        tmp_11 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_12 = in_8 + tmp_11;  in_8 = tmp_11 = None
        tmp_13 = torch.nn.functional.layer_norm(tmp_12, (192,), in_7, in_6, 1e-06);  tmp_12 = in_7 = in_6 = None
        tmp_14 = tmp_13[(slice(None, None, None), 0)]
        tmp_15 = tmp_13[(slice(None, None, None), 1)];  tmp_13 = None
        linear_1 = torch.nn.functional.linear(tmp_14, in_5, in_4);  tmp_14 = in_5 = in_4 = None
        linear_2 = torch.nn.functional.linear(tmp_15, in_3, in_2);  tmp_15 = in_3 = in_2 = None
        tmp_18 = linear_1 + linear_2;  linear_1 = linear_2 = None
        tmp_19 = tmp_18 / 2;  tmp_18 = None
        return (tmp_19,)
        