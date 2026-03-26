import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor):
        linear = torch.nn.functional.linear(w_0, w_2, w_1);  w_0 = w_2 = w_1 = None
        return (linear,)
        