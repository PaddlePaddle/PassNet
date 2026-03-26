import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.silu(in_0, inplace = True);  in_0 = None
        tmp_1 = tmp_0.mean((2, 3))
        tmp_2 = tmp_1.view(1, 1, -1);  tmp_1 = None
        return (tmp_0, tmp_2)
        