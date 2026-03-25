import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, in_0, in_1):
        tmp_6 = torch.nn.functional.gelu(in_1);  in_1 = None
        linear = torch.nn.functional.linear(tmp_6, w_1, w_0);  tmp_6 = w_1 = w_0 = None
        tmp_8 = torch.nn.functional.dropout(linear, 0.1, False, False);  linear = None
        tmp_9 = in_0 + tmp_8;  in_0 = tmp_8 = None
        tmp_10 = torch.nn.functional.layer_norm(tmp_9, (384,), w_3, w_2, 1e-12);  tmp_9 = w_3 = w_2 = None
        tmp_11 = tmp_10[(slice(None, None, None), 0)]
        linear_1 = torch.nn.functional.linear(tmp_11, w_5, w_4);  tmp_11 = w_5 = w_4 = None
        tmp_13 = torch.tanh(linear_1);  linear_1 = None
        return (tmp_10, tmp_13)
        