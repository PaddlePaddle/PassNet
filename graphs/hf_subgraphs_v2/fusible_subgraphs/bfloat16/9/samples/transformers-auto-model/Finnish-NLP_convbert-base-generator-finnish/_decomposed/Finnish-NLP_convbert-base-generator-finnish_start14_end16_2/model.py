import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        linear = torch.nn.functional.linear(in_2, in_1, in_0);  in_1 = in_0 = None
        tmp_3 = in_2.transpose(1, 2);  in_2 = None
        return (linear, tmp_3)
        