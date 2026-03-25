import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor):
        tmp_1 = w_0 * 0.08333333333333333;  w_0 = None
        return (tmp_1,)
        