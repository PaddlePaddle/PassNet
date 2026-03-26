import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor):
        tmp_2 = w_0[in_0];  w_0 = in_0 = None
        return (tmp_2,)
        