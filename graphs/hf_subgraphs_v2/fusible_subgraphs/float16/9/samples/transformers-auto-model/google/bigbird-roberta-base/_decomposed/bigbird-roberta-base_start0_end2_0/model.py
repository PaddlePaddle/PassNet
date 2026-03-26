import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_3 = torch.nn.functional.dropout(in_2, 0.1, False, False);  in_2 = None
        linear = torch.nn.functional.linear(tmp_3, in_1, in_0);  tmp_3 = in_1 = in_0 = None
        return (linear,)
        