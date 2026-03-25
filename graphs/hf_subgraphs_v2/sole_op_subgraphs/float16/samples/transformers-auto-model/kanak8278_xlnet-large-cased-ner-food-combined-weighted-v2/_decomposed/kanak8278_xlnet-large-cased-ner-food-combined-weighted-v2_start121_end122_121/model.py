import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, in_0 : torch.Tensor):
        einsum = torch.functional.einsum('ibh,hnd->ibnd', in_0, w_0);  in_0 = w_0 = None
        return (einsum,)
        