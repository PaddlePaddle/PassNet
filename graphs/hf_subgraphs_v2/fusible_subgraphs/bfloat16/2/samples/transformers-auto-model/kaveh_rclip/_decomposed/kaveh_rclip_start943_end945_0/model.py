import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = in_0.norm(dim = -1, keepdim = True)
        tmp_1 = in_0 / tmp_0;  in_0 = tmp_0 = None
        return (tmp_1,)
        