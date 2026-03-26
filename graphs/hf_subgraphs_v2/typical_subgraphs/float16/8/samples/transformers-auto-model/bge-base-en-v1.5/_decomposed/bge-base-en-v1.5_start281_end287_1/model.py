import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_2 = in_2[(slice(None, None, None), 0)]
        linear = torch.nn.functional.linear(tmp_2, in_1, in_0);  tmp_2 = in_1 = in_0 = None
        tmp_4 = torch.tanh(linear);  linear = tmp_4 = None
        tmp_5 = in_2[(slice(None, None, None), 0)];  in_2 = None
        tmp_6 = torch.cat([tmp_5], 1);  tmp_5 = None
        tmp_7 = torch.nn.functional.normalize(tmp_6, p = 2, dim = 1);  tmp_6 = None
        return (tmp_7,)
        