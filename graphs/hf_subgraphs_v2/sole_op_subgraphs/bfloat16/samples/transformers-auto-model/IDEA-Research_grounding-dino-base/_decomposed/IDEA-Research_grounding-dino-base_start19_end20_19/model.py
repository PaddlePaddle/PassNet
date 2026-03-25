import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.special.logit(in_0, eps = 1e-05);  in_0 = None
        return (tmp_0,)
        