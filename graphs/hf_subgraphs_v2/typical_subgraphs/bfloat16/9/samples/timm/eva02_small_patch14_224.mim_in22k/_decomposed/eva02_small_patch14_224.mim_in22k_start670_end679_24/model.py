import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, in_0, in_1):
        chunk = in_1.chunk(2, dim = -1);  in_1 = None
        tmp_3 = chunk[0]
        tmp_4 = chunk[1];  chunk = None
        tmp_5 = torch.nn.functional.silu(tmp_3, inplace = False);  tmp_3 = None
        tmp_6 = tmp_5 * tmp_4;  tmp_5 = tmp_4 = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False);  tmp_6 = None
        linear = torch.nn.functional.linear(tmp_7, w_1, w_0);  tmp_7 = w_1 = w_0 = None
        tmp_9 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_10 = in_0 + tmp_9;  in_0 = tmp_9 = None
        return (tmp_10,)
        