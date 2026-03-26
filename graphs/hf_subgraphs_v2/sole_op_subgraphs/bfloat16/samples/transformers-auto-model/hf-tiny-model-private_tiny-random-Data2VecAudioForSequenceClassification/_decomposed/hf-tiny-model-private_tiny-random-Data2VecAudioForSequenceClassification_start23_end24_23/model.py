import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.layer_norm(in_0, (16,), None, None, 1e-05);  in_0 = None
        return (tmp_0,)
        