import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, in_0, in_1):
        linear = torch.nn.functional.linear(input = in_0, weight = w_0, bias = in_1);  in_0 = w_0 = in_1 = None
        return (linear,)
        