import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_1 = in_0.unfold(dimension = -1, size = 96, step = 96);  in_0 = None
        return (tmp_1,)
        