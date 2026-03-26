import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        linear = torch.nn.functional.linear(in_0, in_2, in_1);  in_0 = in_2 = in_1 = None
        tmp_4 = in_4.view(1, 20, -1, 64);  in_4 = None
        tmp_5 = tmp_4.transpose(1, 2);  tmp_4 = None
        tmp_6 = linear.view(1, 20, -1, 64);  linear = None
        tmp_7 = tmp_6.transpose(1, 2);  tmp_6 = None
        tmp_8 = in_3[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 20, None))];  in_3 = None
        return (tmp_8, tmp_5, tmp_7)
        