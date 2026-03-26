import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, in_0 : torch.Tensor):
        tmp_4 = in_0[(slice(None, None, None), 0, slice(None, None, None))];  in_0 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, 0.1, False, False);  tmp_4 = None
        linear = torch.nn.functional.linear(tmp_5, w_1, w_0);  tmp_5 = w_1 = w_0 = None
        tmp_7 = torch.tanh(linear);  linear = None
        tmp_8 = torch.nn.functional.dropout(tmp_7, 0.1, False, False);  tmp_7 = None
        linear_1 = torch.nn.functional.linear(tmp_8, w_3, w_2);  tmp_8 = w_3 = w_2 = None
        return (linear_1,)
        