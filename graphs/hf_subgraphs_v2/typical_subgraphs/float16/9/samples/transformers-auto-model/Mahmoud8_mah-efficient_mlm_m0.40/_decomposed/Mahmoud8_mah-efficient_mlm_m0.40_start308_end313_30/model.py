import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, in_0, in_1):
        linear = torch.nn.functional.linear(in_1, w_5, w_4);  in_1 = w_5 = w_4 = None
        tmp_7 = torch.nn.functional.dropout(linear, 0.1, False, False);  linear = None
        tmp_8 = tmp_7 + in_0;  tmp_7 = in_0 = None
        tmp_9 = torch.nn.functional.layer_norm(tmp_8, (1024,), w_1, w_0, 1e-05);  w_1 = w_0 = None
        linear_1 = torch.nn.functional.linear(tmp_9, w_3, w_2);  w_3 = w_2 = None
        return (tmp_8, tmp_9, linear_1)
        