import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        linear = torch.nn.functional.linear(in_7, in_1, in_0);  in_7 = in_1 = in_0 = None
        tmp_7 = torch.nn.functional.dropout(linear, 0.1, False, False);  linear = None
        tmp_8 = tmp_7 + in_6;  tmp_7 = in_6 = None
        tmp_9 = torch.nn.functional.layer_norm(tmp_8, (32,), in_3, in_2, 1e-12);  tmp_8 = in_3 = in_2 = None
        linear_1 = torch.nn.functional.linear(tmp_9, in_5, in_4);  in_5 = in_4 = None
        return (tmp_9, linear_1)
        