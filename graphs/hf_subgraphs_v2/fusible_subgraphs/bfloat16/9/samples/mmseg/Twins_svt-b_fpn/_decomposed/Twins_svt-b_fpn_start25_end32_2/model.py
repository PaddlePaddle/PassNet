import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        linear = torch.nn.functional.linear(in_2, in_1, in_0);  in_2 = in_1 = in_0 = None
        tmp_3 = linear.reshape(1, 361, 49, 3, 3, 32);  linear = None
        tmp_4 = tmp_3.permute(3, 0, 1, 4, 2, 5);  tmp_3 = None
        tmp_5 = tmp_4[0]
        tmp_6 = tmp_4[1]
        tmp_7 = tmp_4[2];  tmp_4 = None
        tmp_8 = tmp_6.transpose(-2, -1);  tmp_6 = None
        return (tmp_5, tmp_8, tmp_7)
        