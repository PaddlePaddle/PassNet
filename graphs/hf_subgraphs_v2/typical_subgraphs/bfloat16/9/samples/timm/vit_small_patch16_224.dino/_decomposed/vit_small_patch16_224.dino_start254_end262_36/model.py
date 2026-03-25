import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, in_0, in_1):
        tmp_4 = torch.nn.functional.gelu(in_1, approximate = 'none');  in_1 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, 0.0, False, False);  tmp_4 = None
        linear = torch.nn.functional.linear(tmp_5, w_1, w_0);  tmp_5 = w_1 = w_0 = None
        tmp_7 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_8 = in_0 + tmp_7;  in_0 = tmp_7 = None
        tmp_9 = torch.nn.functional.layer_norm(tmp_8, (384,), w_3, w_2, 1e-06);  tmp_8 = w_3 = w_2 = None
        tmp_10 = tmp_9[(slice(None, None, None), 0)];  tmp_9 = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, 0.0, False, False);  tmp_10 = None
        return (tmp_11,)
        