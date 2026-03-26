import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        einsum = torch.functional.einsum('bd,bkd->bk', in_1, in_0);  in_1 = in_0 = None
        tmp_1 = einsum.T
        return (einsum, tmp_1)
        