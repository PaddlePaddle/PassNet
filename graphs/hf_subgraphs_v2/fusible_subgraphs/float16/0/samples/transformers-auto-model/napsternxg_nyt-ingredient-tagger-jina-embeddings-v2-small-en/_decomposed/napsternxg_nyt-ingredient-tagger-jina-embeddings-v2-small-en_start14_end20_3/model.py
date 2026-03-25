import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        linear = torch.nn.functional.linear(in_2, in_1, in_0);  in_2 = in_1 = in_0 = None
        tmp_3 = linear.view((1, 64, 8, 64));  linear = None
        tmp_4 = tmp_3.permute(0, 2, 1, 3);  tmp_3 = None
        tmp_5 = in_4.view((1, 64, 8, 64));  in_4 = None
        tmp_6 = tmp_5.permute(0, 2, 1, 3);  tmp_5 = None
        tmp_7 = in_3.transpose(-1, -2);  in_3 = None
        return (tmp_6, tmp_7, tmp_4)
        