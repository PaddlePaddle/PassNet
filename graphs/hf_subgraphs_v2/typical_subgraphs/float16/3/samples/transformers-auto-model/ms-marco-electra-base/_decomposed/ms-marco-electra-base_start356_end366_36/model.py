import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9):
        linear = torch.nn.functional.linear(in_9, in_7, in_6);  in_9 = in_7 = in_6 = None
        tmp_9 = torch.nn.functional.dropout(linear, 0.1, False, False);  linear = None
        tmp_10 = tmp_9 + in_8;  tmp_9 = in_8 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (768,), in_5, in_4, 1e-12);  tmp_10 = in_5 = in_4 = None
        tmp_12 = tmp_11[(slice(None, None, None), 0, slice(None, None, None))];  tmp_11 = None
        tmp_13 = torch.nn.functional.dropout(tmp_12, 0.1, False, False);  tmp_12 = None
        to = tmp_13.to(torch.float16);  tmp_13 = None
        linear_1 = torch.nn.functional.linear(to, in_1, in_0);  to = in_1 = in_0 = None
        tmp_15 = torch.nn.functional.gelu(linear_1);  linear_1 = None
        tmp_16 = torch.nn.functional.dropout(tmp_15, 0.1, False, False);  tmp_15 = None
        to_1 = tmp_16.to(torch.float16);  tmp_16 = None
        linear_2 = torch.nn.functional.linear(to_1, in_3, in_2);  to_1 = in_3 = in_2 = None
        return (linear_2,)
        