import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        linear = torch.nn.functional.linear(in_2, in_1, in_0);  in_2 = in_1 = in_0 = None
        tmp_3 = torch.nn.functional.dropout(linear, 0.1, False, False);  linear = None
        tmp_4 = tmp_3.transpose(-2, -1)
        return (tmp_3, tmp_4)
        