import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        linear = torch.nn.functional.linear(in_0, in_1, None);  in_0 = in_1 = None
        tmp_3 = in_2.view(2, -1, 16, 64);  in_2 = None
        tmp_4 = tmp_3.transpose(1, 2);  tmp_3 = None
        tmp_5 = linear.view(2, -1, 16, 64);  linear = None
        tmp_6 = tmp_5.transpose(1, 2);  tmp_5 = None
        tmp_7 = tmp_4.transpose(3, 2)
        return (tmp_4, tmp_7, tmp_6)
        