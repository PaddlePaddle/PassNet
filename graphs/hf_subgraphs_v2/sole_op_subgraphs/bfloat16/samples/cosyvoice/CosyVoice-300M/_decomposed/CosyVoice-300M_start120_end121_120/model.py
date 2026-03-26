import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor):
        linear = torch.nn.functional.linear(in_0, w_0, None);  in_0 = w_0 = None
        return (linear,)
        