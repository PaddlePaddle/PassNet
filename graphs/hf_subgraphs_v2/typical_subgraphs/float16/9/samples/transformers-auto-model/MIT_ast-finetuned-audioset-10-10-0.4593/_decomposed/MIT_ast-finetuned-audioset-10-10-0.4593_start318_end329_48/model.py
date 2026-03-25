import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, in_0, in_1):
        tmp_8 = torch.nn.functional.gelu(in_1);  in_1 = None
        linear = torch.nn.functional.linear(tmp_8, w_1, w_0);  tmp_8 = w_1 = w_0 = None
        tmp_10 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_11 = tmp_10 + in_0;  tmp_10 = in_0 = None
        tmp_12 = torch.nn.functional.layer_norm(tmp_11, (768,), w_3, w_2, 1e-12);  tmp_11 = w_3 = w_2 = None
        tmp_13 = tmp_12[(slice(None, None, None), 0)]
        tmp_14 = tmp_12[(slice(None, None, None), 1)];  tmp_12 = None
        tmp_15 = tmp_13 + tmp_14;  tmp_13 = tmp_14 = None
        tmp_16 = tmp_15 / 2;  tmp_15 = None
        tmp_17 = torch.nn.functional.layer_norm(tmp_16, (768,), w_7, w_6, 1e-12);  tmp_16 = w_7 = w_6 = None
        linear_1 = torch.nn.functional.linear(tmp_17, w_5, w_4);  tmp_17 = w_5 = w_4 = None
        return (linear_1,)
        