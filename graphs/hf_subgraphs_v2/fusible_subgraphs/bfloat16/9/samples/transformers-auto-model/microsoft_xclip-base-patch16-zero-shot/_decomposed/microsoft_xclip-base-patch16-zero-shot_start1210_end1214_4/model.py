import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        linear = torch.nn.functional.linear(in_2, in_0, None);  in_2 = in_0 = None
        tmp_2 = linear.reshape(1, 196, 8, 64);  linear = None
        tmp_3 = tmp_2.permute(0, 2, 1, 3);  tmp_2 = None
        tmp_4 = in_1.transpose(-2, -1);  in_1 = None
        return (tmp_4, tmp_3)
        