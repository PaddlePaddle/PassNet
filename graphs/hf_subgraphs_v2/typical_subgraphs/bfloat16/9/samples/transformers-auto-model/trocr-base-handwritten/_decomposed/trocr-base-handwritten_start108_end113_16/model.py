import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, in_0, in_1):
        linear = torch.nn.functional.linear(in_1, w_1, w_0);  in_1 = w_1 = w_0 = None
        tmp_6 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_7 = tmp_6 + in_0;  tmp_6 = in_0 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (768,), w_4, w_3, 1e-12);  w_4 = w_3 = None
        linear_1 = torch.nn.functional.linear(tmp_8, w_2, None);  w_2 = None
        return (tmp_7, tmp_8, linear_1)
        