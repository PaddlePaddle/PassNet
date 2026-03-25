import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        einsum = torch.functional.einsum('bck,bnk->bcn', in_1, in_0);  in_1 = in_0 = None
        tmp_1 = einsum.view(2, 512, 64, 64);  einsum = None
        return (tmp_1,)
        