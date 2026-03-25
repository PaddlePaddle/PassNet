import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        linear = torch.nn.functional.linear(in_2, in_1, in_0);  in_2 = in_1 = in_0 = None
        tmp_4 = torch.nn.functional.dropout(linear, p = 0.0, training = False);  linear = None
        tmp_5 = in_3 + tmp_4;  in_3 = None
        return (tmp_5, tmp_4)
        