import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = in_0.repeat(1, 1, 1, 1);  in_0 = None
        return (tmp_0,)
        