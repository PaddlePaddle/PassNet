import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_0 = in_0.masked_fill(in_1, 0.0);  in_0 = in_1 = None
        return (tmp_0,)
        