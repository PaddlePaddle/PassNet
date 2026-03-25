import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, in_0, in_1):
        tmp_2 = torch.nn.functional.gelu(in_1, approximate = 'none');  in_1 = None
        tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False);  tmp_2 = None
        linear = torch.nn.functional.linear(tmp_3, w_1, w_0);  tmp_3 = w_1 = w_0 = None
        tmp_5 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_6 = in_0 + tmp_5;  in_0 = tmp_5 = None
        return (tmp_6,)
        