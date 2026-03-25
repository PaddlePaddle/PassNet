import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, in_0, in_1):
        tmp_6 = torch.nn.functional.gelu(in_1, approximate = 'none');  in_1 = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False);  tmp_6 = None
        linear = torch.nn.functional.linear(tmp_7, w_1, w_0);  tmp_7 = w_1 = w_0 = None
        tmp_9 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_10 = in_0 + tmp_9;  in_0 = tmp_9 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (1280,), w_5, w_4, 1e-05);  tmp_10 = w_5 = w_4 = None
        tmp_12 = tmp_11[(slice(None, None, None), 0)];  tmp_11 = None
        tmp_13 = torch.nn.functional.dropout(tmp_12, 0.0, False, False);  tmp_12 = None
        linear_1 = torch.nn.functional.linear(tmp_13, w_3, w_2);  tmp_13 = w_3 = w_2 = None
        return (linear_1,)
        