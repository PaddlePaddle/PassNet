import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, in_0 : torch.Tensor):
        tmp_2 = in_0[(slice(None, None, None), 0)]
        linear = torch.nn.functional.linear(tmp_2, w_1, w_0);  tmp_2 = w_1 = w_0 = None
        tmp_4 = torch.tanh(linear);  linear = tmp_4 = None
        tmp_5 = in_0[(slice(None, None, None), 0)];  in_0 = None
        tmp_6 = torch.cat([tmp_5], 1);  tmp_5 = None
        tmp_7 = torch.nn.functional.normalize(tmp_6, p = 2, dim = 1);  tmp_6 = None
        return (tmp_7,)
        