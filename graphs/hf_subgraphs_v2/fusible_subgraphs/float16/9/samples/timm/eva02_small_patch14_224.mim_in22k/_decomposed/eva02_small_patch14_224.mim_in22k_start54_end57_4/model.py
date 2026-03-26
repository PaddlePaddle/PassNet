import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        chunk = in_0.chunk(2, dim = -1);  in_0 = None
        tmp_1 = chunk[0]
        tmp_2 = chunk[1];  chunk = None
        return (tmp_1, tmp_2)
        