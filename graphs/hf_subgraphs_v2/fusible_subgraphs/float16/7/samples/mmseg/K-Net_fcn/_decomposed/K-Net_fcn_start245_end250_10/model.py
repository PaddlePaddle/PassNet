import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        linear = torch.nn.functional.linear(in_2, in_1, in_0);  in_2 = in_1 = in_0 = None
        tmp_3 = linear.permute(0, 1, 3, 2);  linear = None
        tmp_4 = tmp_3.reshape(1, 150, 512, 1, 1);  tmp_3 = None
        tmp_5 = in_3[slice(0, 1, None)];  in_3 = None
        tmp_6 = tmp_4[0];  tmp_4 = None
        return (tmp_5, tmp_6)
        