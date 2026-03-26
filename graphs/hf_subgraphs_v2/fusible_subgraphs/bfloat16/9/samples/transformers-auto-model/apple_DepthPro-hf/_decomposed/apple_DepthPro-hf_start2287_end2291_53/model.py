import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        linear = torch.nn.functional.linear(in_2, in_1, in_0);  in_2 = in_1 = in_0 = None
        tmp_3 = linear[(slice(None, None, None), slice(-576, None, None), slice(None, None, None))];  linear = None
        tmp_4 = tmp_3.reshape(1, 24, 24, 128);  tmp_3 = None
        tmp_5 = tmp_4.permute(0, 3, 1, 2);  tmp_4 = None
        return (tmp_5,)
        