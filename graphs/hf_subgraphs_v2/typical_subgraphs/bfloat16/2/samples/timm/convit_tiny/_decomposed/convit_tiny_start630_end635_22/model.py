import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6):
        linear = torch.nn.functional.linear(in_6, in_1, in_0);  in_6 = in_1 = in_0 = None
        tmp_6 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_7 = in_5 + tmp_6;  in_5 = tmp_6 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (192,), in_4, in_3, 1e-06);  in_4 = in_3 = None
        linear_1 = torch.nn.functional.linear(tmp_8, in_2, None);  tmp_8 = in_2 = None
        return (linear_1, tmp_7)
        