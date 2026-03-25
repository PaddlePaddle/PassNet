import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor):
        tmp_2 = torch._weight_norm(w_1, w_0, 2);  w_1 = w_0 = None
        return (tmp_2,)
        