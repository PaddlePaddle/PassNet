import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        linear = torch.nn.functional.linear(in_1, in_0, None);  in_1 = in_0 = None
        tmp_2 = linear.view((32, 64, -1, 128));  linear = None
        tmp_3 = tmp_2.transpose(1, 2);  tmp_2 = None
        tmp_4 = in_2[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None))];  in_2 = None
        tmp_5 = tmp_4.expand(32, 4, 4, 64, 128);  tmp_4 = None
        return (tmp_5, tmp_3)
        