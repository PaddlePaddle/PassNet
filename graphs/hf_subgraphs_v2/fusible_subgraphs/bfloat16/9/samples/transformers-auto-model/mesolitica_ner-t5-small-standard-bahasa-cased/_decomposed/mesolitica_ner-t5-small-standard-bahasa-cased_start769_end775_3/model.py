import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        linear = torch.nn.functional.linear(in_1, in_0, None);  in_1 = in_0 = None
        tmp_2 = in_2.view(1, -1, 8, 64);  in_2 = None
        tmp_3 = tmp_2.transpose(1, 2);  tmp_2 = None
        tmp_4 = linear.view(1, -1, 8, 64);  linear = None
        tmp_5 = tmp_4.transpose(1, 2);  tmp_4 = None
        tmp_6 = tmp_3.transpose(3, 2);  tmp_3 = None
        return (tmp_6, tmp_5)
        