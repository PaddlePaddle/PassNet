import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        linear = torch.nn.functional.linear(in_3, in_1, in_0);  in_3 = in_1 = in_0 = None
        tmp_3 = linear.permute(0, 3, 1, 2);  linear = None
        tmp_4 = in_2.transpose(-2, -1);  in_2 = None
        return (tmp_3, tmp_4)
        