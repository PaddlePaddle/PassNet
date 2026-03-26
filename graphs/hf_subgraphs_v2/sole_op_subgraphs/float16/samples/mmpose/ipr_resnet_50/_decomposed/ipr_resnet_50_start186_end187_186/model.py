import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, in_0 : torch.Tensor):
        tmp_1 = in_0.mul(w_0);  in_0 = w_0 = None
        return (tmp_1,)
        