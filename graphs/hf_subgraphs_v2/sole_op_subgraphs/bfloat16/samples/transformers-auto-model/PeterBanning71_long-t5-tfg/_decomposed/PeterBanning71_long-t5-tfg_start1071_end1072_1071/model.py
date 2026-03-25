import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        einsum = torch.functional.einsum('...nd,...ng->...gd', in_0, in_1);  in_0 = in_1 = None
        return (einsum,)
        