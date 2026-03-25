import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, in_0, in_1):
        tmp_6 = torch.nn.functional.gelu(in_1);  in_1 = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False);  tmp_6 = None
        linear = torch.nn.functional.linear(tmp_7, w_1, w_0);  tmp_7 = w_1 = w_0 = None
        tmp_9 = torch.nn.functional.dropout(linear, 0.1, False, False);  linear = None
        tmp_10 = in_0 + tmp_9;  in_0 = tmp_9 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (32,), w_3, w_2, 1e-05);  tmp_10 = w_3 = w_2 = None
        tmp_12 = tmp_11[(slice(None, None, None), 0)]
        linear_1 = torch.nn.functional.linear(tmp_12, w_5, w_4);  tmp_12 = w_5 = w_4 = None
        tmp_14 = torch.tanh(linear_1);  linear_1 = None
        return (tmp_11, tmp_14)
        