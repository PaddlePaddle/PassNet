import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor):
        linear = torch.nn.functional.linear(in_1, weight = w_0, bias = in_0);  in_1 = w_0 = in_0 = None
        return (linear,)
        