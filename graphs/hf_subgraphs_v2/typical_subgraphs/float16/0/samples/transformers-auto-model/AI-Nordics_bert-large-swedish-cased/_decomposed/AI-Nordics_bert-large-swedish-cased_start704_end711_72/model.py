import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        linear = torch.nn.functional.linear(in_7, in_1, in_0);  in_7 = in_1 = in_0 = None
        tmp_7 = torch.nn.functional.dropout(linear, 0.1, False, False);  linear = None
        tmp_8 = in_6 + tmp_7;  in_6 = tmp_7 = None
        tmp_9 = torch.nn.functional.layer_norm(tmp_8, (1024,), in_3, in_2, 1e-12);  tmp_8 = in_3 = in_2 = None
        tmp_10 = tmp_9[(slice(None, None, None), 0)]
        linear_1 = torch.nn.functional.linear(tmp_10, in_5, in_4);  tmp_10 = in_5 = in_4 = None
        tmp_12 = torch.tanh(linear_1);  linear_1 = None
        return (tmp_9, tmp_12)
        