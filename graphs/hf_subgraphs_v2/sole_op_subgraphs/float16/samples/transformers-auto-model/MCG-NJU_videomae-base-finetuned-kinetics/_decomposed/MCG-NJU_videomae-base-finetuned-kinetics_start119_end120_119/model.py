import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, in_0 : torch.Tensor):
        linear = torch.nn.functional.linear(input = in_0, weight = w_0, bias = w_1);  in_0 = w_0 = w_1 = None
        return (linear,)
        