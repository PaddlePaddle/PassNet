import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        tmp_3 = in_3[(slice(None, None, None), 0)];  in_3 = None
        to = tmp_3.to(torch.bfloat16);  tmp_3 = None
        linear = torch.nn.functional.linear(to, in_2, in_1);  to = in_2 = in_1 = None
        tmp_5 = torch.tanh(linear);  linear = tmp_5 = None
        tmp_6 = in_0.unsqueeze(-1);  in_0 = None
        tmp_7 = tmp_6.expand((1, 10, 1024));  tmp_6 = None
        return (tmp_7,)
        