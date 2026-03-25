import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        linear = torch.nn.functional.linear(in_1, in_0, None);  in_1 = in_0 = None
        tmp_2 = linear.reshape(1, 197, 12, -1);  linear = None
        tmp_3 = tmp_2.transpose(1, 2);  tmp_2 = None
        return (tmp_3,)
        