import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        linear = torch.nn.functional.linear(in_2, in_1, in_0);  in_2 = in_1 = in_0 = None
        tmp_3 = linear.reshape(6, 49, 3, 4, 32);  linear = None
        tmp_4 = tmp_3.permute(2, 0, 3, 1, 4);  tmp_3 = None
        tmp_5 = tmp_4[0]
        tmp_6 = tmp_4[1]
        tmp_7 = tmp_4[2];  tmp_4 = None
        return (tmp_6, tmp_5, tmp_7)
        