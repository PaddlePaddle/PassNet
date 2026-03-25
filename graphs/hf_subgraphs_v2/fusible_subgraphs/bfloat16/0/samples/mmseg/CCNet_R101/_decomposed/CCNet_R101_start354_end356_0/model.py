import torch

from torch import inf

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        einsum = torch.functional.einsum('bchw,bciw->bwhi', in_1, in_0);  in_1 = in_0 = None
        tmp_1 = torch.tensor(-inf)
        return (einsum, tmp_1)
        