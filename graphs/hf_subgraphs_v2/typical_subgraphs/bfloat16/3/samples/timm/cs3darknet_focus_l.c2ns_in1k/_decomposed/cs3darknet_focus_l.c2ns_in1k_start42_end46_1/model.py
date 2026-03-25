import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.silu(in_0, inplace = True);  in_0 = None
        split = tmp_0.split(128, dim = 1);  tmp_0 = None
        tmp_2 = split[0]
        tmp_3 = split[1];  split = None
        return (tmp_2, tmp_3)
        