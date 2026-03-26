import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        linear = torch.nn.functional.linear(in_3, in_1, in_0);  in_3 = in_1 = in_0 = None
        tmp_3 = in_4.view(1, 50, -1, 64);  in_4 = None
        tmp_4 = tmp_3.transpose(1, 2);  tmp_3 = None
        tmp_5 = linear.view(1, 50, -1, 64);  linear = None
        tmp_6 = tmp_5.transpose(1, 2);  tmp_5 = None
        tmp_7 = in_2[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 50, None))];  in_2 = None
        return (tmp_7, tmp_4, tmp_6)
        