import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.mean(in_0, dim = 1, keepdim = True);  in_0 = None
        return (tmp_0,)
        