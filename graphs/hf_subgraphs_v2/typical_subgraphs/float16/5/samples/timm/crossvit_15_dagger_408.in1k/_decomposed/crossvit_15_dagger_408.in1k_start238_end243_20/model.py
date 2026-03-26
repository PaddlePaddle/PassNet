import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8):
        linear = torch.nn.functional.linear(in_8, in_1, in_0);  in_8 = in_1 = in_0 = None
        tmp_7 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_8 = in_7 + tmp_7;  in_7 = tmp_7 = None
        tmp_9 = torch.nn.functional.layer_norm(in_6, (384,), in_5, in_4, 1e-06);  in_6 = in_5 = in_4 = None
        linear_1 = torch.nn.functional.linear(tmp_9, in_3, in_2);  tmp_9 = in_3 = in_2 = None
        return (linear_1, tmp_8)
        