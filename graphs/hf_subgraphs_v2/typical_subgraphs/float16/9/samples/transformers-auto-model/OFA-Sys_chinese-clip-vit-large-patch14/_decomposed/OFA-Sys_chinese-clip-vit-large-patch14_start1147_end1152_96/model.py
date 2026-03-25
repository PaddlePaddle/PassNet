import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, in_0, in_1, in_2):
        linear = torch.nn.functional.linear(in_1, w_3, w_2);  in_1 = w_3 = w_2 = None
        tmp_6 = torch.nn.functional.dropout(linear, 0.1, False, False);  linear = None
        tmp_7 = tmp_6 + in_0;  tmp_6 = in_0 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (768,), w_1, w_0, 1e-12);  tmp_7 = w_1 = w_0 = None
        linear_1 = torch.nn.functional.linear(in_2, w_4, None);  in_2 = w_4 = None
        return (tmp_8, linear_1)
        