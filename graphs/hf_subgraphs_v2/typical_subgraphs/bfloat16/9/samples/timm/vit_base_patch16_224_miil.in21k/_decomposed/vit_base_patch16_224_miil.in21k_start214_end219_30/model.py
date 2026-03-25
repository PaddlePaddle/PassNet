import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, in_0, in_1):
        linear = torch.nn.functional.linear(in_1, w_4, w_3);  in_1 = w_4 = w_3 = None
        tmp_6 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_7 = in_0 + tmp_6;  in_0 = tmp_6 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (768,), w_2, w_1, 1e-06);  w_2 = w_1 = None
        linear_1 = torch.nn.functional.linear(tmp_8, w_0, None);  tmp_8 = w_0 = None
        return (linear_1, tmp_7)
        