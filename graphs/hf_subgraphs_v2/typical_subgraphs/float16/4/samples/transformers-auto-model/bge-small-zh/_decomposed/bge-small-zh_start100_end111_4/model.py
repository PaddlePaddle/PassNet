import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_6 = torch.nn.functional.gelu(in_7);  in_7 = None
        linear = torch.nn.functional.linear(tmp_6, in_3, in_2);  tmp_6 = in_3 = in_2 = None
        tmp_8 = torch.nn.functional.dropout(linear, 0.1, False, False);  linear = None
        tmp_9 = tmp_8 + in_6;  tmp_8 = in_6 = None
        tmp_10 = torch.nn.functional.layer_norm(tmp_9, (512,), in_1, in_0, 1e-12);  tmp_9 = in_1 = in_0 = None
        tmp_11 = tmp_10[(slice(None, None, None), 0)]
        linear_1 = torch.nn.functional.linear(tmp_11, in_5, in_4);  tmp_11 = in_5 = in_4 = None
        tmp_13 = torch.tanh(linear_1);  linear_1 = tmp_13 = None
        tmp_14 = tmp_10[(slice(None, None, None), 0)]
        tmp_15 = torch.cat([tmp_14], 1);  tmp_14 = None
        tmp_16 = torch.nn.functional.normalize(tmp_15, p = 2, dim = 1);  tmp_15 = None
        return (tmp_10, tmp_16)
        