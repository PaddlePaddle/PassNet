import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, in_0 : torch.Tensor):
        tmp_1 = torch.nn.functional.layer_norm(in_0, (64,), w_0, None, 1e-06);  in_0 = w_0 = None
        return (tmp_1,)
        