import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9):
        tmp_8 = torch.nn.functional.gelu(in_9);  in_9 = None
        to = tmp_8.to(torch.float16);  tmp_8 = None
        linear = torch.nn.functional.linear(to, in_3, in_2);  to = in_3 = in_2 = None
        tmp_10 = torch.nn.functional.dropout(linear, 0.1, False, False);  linear = None
        tmp_11 = tmp_10 + in_8;  tmp_10 = in_8 = None
        tmp_12 = torch.nn.functional.layer_norm(tmp_11, (128,), in_1, in_0, 1e-12);  tmp_11 = in_1 = in_0 = None
        tmp_13 = tmp_12[(slice(None, None, None), 0)];  tmp_12 = None
        to_1 = tmp_13.to(torch.float16);  tmp_13 = None
        linear_1 = torch.nn.functional.linear(to_1, in_5, in_4);  to_1 = in_5 = in_4 = None
        tmp_15 = torch.tanh(linear_1);  linear_1 = None
        tmp_16 = torch.nn.functional.dropout(tmp_15, 0.1, False, False);  tmp_15 = None
        to_2 = tmp_16.to(torch.float16);  tmp_16 = None
        linear_2 = torch.nn.functional.linear(to_2, in_7, in_6);  to_2 = in_7 = in_6 = None
        return (linear_2,)
        