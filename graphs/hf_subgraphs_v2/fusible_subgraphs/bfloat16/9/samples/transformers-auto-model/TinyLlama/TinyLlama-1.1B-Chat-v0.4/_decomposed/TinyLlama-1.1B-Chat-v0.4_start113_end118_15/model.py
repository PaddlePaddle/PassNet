import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        linear = torch.nn.functional.linear(in_2, in_0, None);  in_2 = in_0 = None
        tmp_2 = linear.view((1, 3, -1, 64));  linear = None
        tmp_3 = tmp_2.transpose(1, 2);  tmp_2 = None
        tmp_4 = in_1.unsqueeze(1);  in_1 = None
        tmp_5 = in_3.unsqueeze(1);  in_3 = None
        return (tmp_4, tmp_5, tmp_3)
        