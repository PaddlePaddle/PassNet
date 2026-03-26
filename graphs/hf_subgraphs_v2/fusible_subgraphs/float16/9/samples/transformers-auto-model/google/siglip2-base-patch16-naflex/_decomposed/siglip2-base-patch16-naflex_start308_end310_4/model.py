import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_1 = in_0.norm(p = 2, dim = -1, keepdim = True)
        tmp_2 = in_0 / tmp_1;  in_0 = tmp_1 = None
        return (tmp_2,)
        