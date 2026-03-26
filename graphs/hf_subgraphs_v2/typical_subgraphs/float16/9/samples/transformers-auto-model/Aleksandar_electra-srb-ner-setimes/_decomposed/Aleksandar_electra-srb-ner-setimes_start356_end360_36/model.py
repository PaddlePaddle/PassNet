import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, in_0, in_1):
        linear = torch.nn.functional.linear(in_1, w_3, w_2);  in_1 = w_3 = w_2 = None
        tmp_5 = torch.nn.functional.dropout(linear, 0.1, False, False);  linear = None
        tmp_6 = tmp_5 + in_0;  tmp_5 = in_0 = None
        tmp_7 = torch.nn.functional.layer_norm(tmp_6, (768,), w_1, w_0, 1e-12);  tmp_6 = w_1 = w_0 = None
        return (tmp_7,)
        