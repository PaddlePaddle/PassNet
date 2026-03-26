import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor):
        tmp_1 = w_0[slice(None, 2209, None)];  w_0 = None
        return (tmp_1,)
        