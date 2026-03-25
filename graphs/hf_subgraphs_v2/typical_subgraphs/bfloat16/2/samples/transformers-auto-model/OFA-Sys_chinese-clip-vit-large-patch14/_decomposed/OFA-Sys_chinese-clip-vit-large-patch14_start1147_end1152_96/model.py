import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        linear = torch.nn.functional.linear(in_6, in_3, in_2);  in_6 = in_3 = in_2 = None
        tmp_6 = torch.nn.functional.dropout(linear, 0.1, False, False);  linear = None
        tmp_7 = tmp_6 + in_5;  tmp_6 = in_5 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (768,), in_1, in_0, 1e-12);  tmp_7 = in_1 = in_0 = None
        linear_1 = torch.nn.functional.linear(in_7, in_4, None);  in_7 = in_4 = None
        return (tmp_8, linear_1)
        