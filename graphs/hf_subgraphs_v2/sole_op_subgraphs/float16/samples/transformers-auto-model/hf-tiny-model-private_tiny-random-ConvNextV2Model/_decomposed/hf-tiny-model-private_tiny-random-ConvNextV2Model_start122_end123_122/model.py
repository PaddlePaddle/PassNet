import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.linalg.vector_norm(in_0, ord = 2, dim = (1, 2), keepdim = True);  in_0 = None
        return (tmp_0,)
        