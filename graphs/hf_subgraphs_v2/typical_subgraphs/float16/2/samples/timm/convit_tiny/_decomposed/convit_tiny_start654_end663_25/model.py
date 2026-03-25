import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_6 = torch.nn.functional.gelu(in_7, approximate = 'none');  in_7 = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False);  tmp_6 = None
        linear = torch.nn.functional.linear(tmp_7, in_1, in_0);  tmp_7 = in_1 = in_0 = None
        tmp_9 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_10 = in_6 + tmp_9;  in_6 = tmp_9 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (192,), in_5, in_4, 1e-06);  tmp_10 = in_5 = in_4 = None
        tmp_12 = tmp_11[(slice(None, None, None), 0)];  tmp_11 = None
        tmp_13 = torch.nn.functional.dropout(tmp_12, 0.0, False, False);  tmp_12 = None
        linear_1 = torch.nn.functional.linear(tmp_13, in_3, in_2);  tmp_13 = in_3 = in_2 = None
        return (linear_1,)
        