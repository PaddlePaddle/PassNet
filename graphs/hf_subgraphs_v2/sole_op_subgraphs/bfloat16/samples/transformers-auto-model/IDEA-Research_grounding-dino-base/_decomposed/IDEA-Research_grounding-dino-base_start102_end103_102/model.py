import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = in_1.masked_fill(in_0, -1000000.0);  in_1 = in_0 = None
        return (tmp_0,)
        