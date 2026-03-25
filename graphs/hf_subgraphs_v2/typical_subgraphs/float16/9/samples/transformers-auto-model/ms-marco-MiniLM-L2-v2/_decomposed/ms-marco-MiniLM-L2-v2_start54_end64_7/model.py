import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, in_0, in_1):
        tmp_8 = torch.nn.functional.gelu(in_1);  in_1 = None
        linear = torch.nn.functional.linear(tmp_8, w_3, w_2);  tmp_8 = w_3 = w_2 = None
        tmp_10 = torch.nn.functional.dropout(linear, 0.1, False, False);  linear = None
        tmp_11 = tmp_10 + in_0;  tmp_10 = in_0 = None
        tmp_12 = torch.nn.functional.layer_norm(tmp_11, (384,), w_1, w_0, 1e-12);  tmp_11 = w_1 = w_0 = None
        tmp_13 = tmp_12[(slice(None, None, None), 0)];  tmp_12 = None
        linear_1 = torch.nn.functional.linear(tmp_13, w_5, w_4);  tmp_13 = w_5 = w_4 = None
        tmp_15 = torch.tanh(linear_1);  linear_1 = None
        tmp_16 = torch.nn.functional.dropout(tmp_15, 0.1, False, False);  tmp_15 = None
        linear_2 = torch.nn.functional.linear(tmp_16, w_7, w_6);  tmp_16 = w_7 = w_6 = None
        return (linear_2,)
        