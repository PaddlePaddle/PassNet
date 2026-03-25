import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, in_0, in_1):
        linear = torch.nn.functional.linear(in_1, w_1, w_0);  in_1 = w_1 = w_0 = None
        tmp_7 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_8 = in_0 + tmp_7;  in_0 = tmp_7 = None
        tmp_9 = torch.nn.functional.layer_norm(tmp_8, (256,), w_5, w_4, 1e-05);  tmp_8 = w_5 = w_4 = None
        tmp_10 = tmp_9[(slice(None, None, None), 0)];  tmp_9 = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, 0.0, False, False);  tmp_10 = None
        linear_1 = torch.nn.functional.linear(tmp_11, w_3, w_2);  tmp_11 = w_3 = w_2 = None
        return (linear_1,)
        