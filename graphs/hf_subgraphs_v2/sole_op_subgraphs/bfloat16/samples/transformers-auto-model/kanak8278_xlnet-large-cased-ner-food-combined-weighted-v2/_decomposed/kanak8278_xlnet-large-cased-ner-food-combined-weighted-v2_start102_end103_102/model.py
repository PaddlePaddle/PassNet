import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        einsum = torch.functional.einsum('ijbn->bnij', in_0);  in_0 = None
        return (einsum,)
        