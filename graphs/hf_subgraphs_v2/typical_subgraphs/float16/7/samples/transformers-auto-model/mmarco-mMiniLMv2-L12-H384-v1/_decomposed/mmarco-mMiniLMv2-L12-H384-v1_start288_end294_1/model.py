import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        tmp_4 = in_4[(slice(None, None, None), 0, slice(None, None, None))];  in_4 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, 0.1, False, False);  tmp_4 = None
        to = tmp_5.to(torch.float16);  tmp_5 = None
        linear = torch.nn.functional.linear(to, in_1, in_0);  to = in_1 = in_0 = None
        tmp_7 = torch.tanh(linear);  linear = None
        tmp_8 = torch.nn.functional.dropout(tmp_7, 0.1, False, False);  tmp_7 = None
        to_1 = tmp_8.to(torch.float16);  tmp_8 = None
        linear_1 = torch.nn.functional.linear(to_1, in_3, in_2);  to_1 = in_3 = in_2 = None
        return (linear_1,)
        