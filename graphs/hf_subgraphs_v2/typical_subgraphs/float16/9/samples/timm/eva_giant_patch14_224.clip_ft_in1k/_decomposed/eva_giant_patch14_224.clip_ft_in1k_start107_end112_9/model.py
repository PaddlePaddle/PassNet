import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, in_0, in_1):
        linear = torch.nn.functional.linear(in_1, w_1, w_0);  in_1 = w_1 = w_0 = None
        tmp_7 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_8 = in_0 + tmp_7;  in_0 = tmp_7 = None
        tmp_9 = torch.nn.functional.layer_norm(tmp_8, (1408,), w_5, w_4, 1e-06);  w_5 = w_4 = None
        linear_1 = torch.nn.functional.linear(tmp_9, w_3, w_2);  tmp_9 = w_3 = w_2 = None
        return (tmp_8, linear_1)
        