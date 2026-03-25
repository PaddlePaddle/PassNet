import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8):
        linear = torch.nn.functional.linear(in_7, in_1, in_0);  in_7 = in_1 = in_0 = None
        tmp_7 = torch.nn.functional.silu(in_8, inplace = False);  in_8 = None
        tmp_8 = tmp_7 * linear;  tmp_7 = linear = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False);  tmp_8 = None
        tmp_10 = torch.nn.functional.layer_norm(tmp_9, (2730,), in_5, in_4, 1e-06);  tmp_9 = in_5 = in_4 = None
        linear_1 = torch.nn.functional.linear(tmp_10, in_3, in_2);  tmp_10 = in_3 = in_2 = None
        tmp_12 = torch.nn.functional.dropout(linear_1, 0.0, False, False);  linear_1 = None
        tmp_13 = in_6 + tmp_12;  in_6 = tmp_12 = None
        return (tmp_13,)
        