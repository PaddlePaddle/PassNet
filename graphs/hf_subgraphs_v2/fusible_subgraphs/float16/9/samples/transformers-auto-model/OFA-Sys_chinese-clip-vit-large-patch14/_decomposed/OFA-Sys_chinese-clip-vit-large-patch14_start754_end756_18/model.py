import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        tmp_3 = torch.nn.functional.layer_norm(in_3, (1024,), in_2, in_1, 1e-05);  in_3 = in_2 = in_1 = None
        tmp_4 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        return (tmp_4, tmp_3)
        