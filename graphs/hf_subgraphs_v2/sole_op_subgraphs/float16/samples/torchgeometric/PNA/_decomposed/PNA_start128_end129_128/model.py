import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        tmp_0 = in_2.scatter_add_(0, in_0, in_1);  in_2 = in_0 = in_1 = None
        return (tmp_0,)
        