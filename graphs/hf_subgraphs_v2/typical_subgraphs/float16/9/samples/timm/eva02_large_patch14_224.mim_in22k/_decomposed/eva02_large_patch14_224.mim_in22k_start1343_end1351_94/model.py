import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, in_0, in_1, in_2):
        linear = torch.nn.functional.linear(in_1, w_1, w_0);  in_1 = w_1 = w_0 = None
        tmp_7 = torch.nn.functional.silu(in_2, inplace = False);  in_2 = None
        tmp_8 = tmp_7 * linear;  tmp_7 = linear = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False);  tmp_8 = None
        tmp_10 = torch.nn.functional.layer_norm(tmp_9, (2730,), w_5, w_4, 1e-06);  tmp_9 = w_5 = w_4 = None
        linear_1 = torch.nn.functional.linear(tmp_10, w_3, w_2);  tmp_10 = w_3 = w_2 = None
        tmp_12 = torch.nn.functional.dropout(linear_1, 0.0, False, False);  linear_1 = None
        tmp_13 = in_0 + tmp_12;  in_0 = tmp_12 = None
        return (tmp_13,)
        