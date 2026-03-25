import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_2 = in_2[(slice(None, None, None), 0)]
        to = tmp_2.to(torch.bfloat16);  tmp_2 = None
        linear = torch.nn.functional.linear(to, in_1, in_0);  to = in_1 = in_0 = None
        tmp_4 = torch.tanh(linear);  linear = tmp_4 = None
        tmp_5 = in_2[(slice(None, None, None), 0)];  in_2 = None
        return (tmp_5,)
        