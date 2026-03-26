import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_2 = torch.nn.functional.layer_norm(in_2, (2,), in_1, in_0, 1e-12);  in_2 = in_1 = in_0 = None
        tmp_3 = tmp_2[(slice(None, None, None), 0)]
        return (tmp_3, tmp_2)
        