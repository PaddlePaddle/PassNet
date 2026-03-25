import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        einsum = torch.functional.einsum('bciw,bhwi->bchw', in_2, in_1);  in_2 = in_1 = None
        tmp_1 = in_0[(Ellipsis, slice(64, None, None))];  in_0 = None
        return (tmp_1, einsum)
        