import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_2 = torch.cat([in_0, in_1], dim = -1);  in_0 = in_1 = None
        return (tmp_2,)
        