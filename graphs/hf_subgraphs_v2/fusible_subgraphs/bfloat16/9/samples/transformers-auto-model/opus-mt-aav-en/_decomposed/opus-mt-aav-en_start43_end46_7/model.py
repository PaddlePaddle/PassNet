import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        linear = torch.nn.functional.linear(in_2, in_1, in_0);  in_2 = in_1 = in_0 = None
        tmp_3 = linear.view(1, 29, -1, 64);  linear = None
        tmp_4 = tmp_3.transpose(1, 2);  tmp_3 = None
        return (tmp_4,)
        