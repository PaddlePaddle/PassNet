import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_1 = torch.nn.functional.layer_norm(in_0, (1, 64000));  in_0 = None
        return (tmp_1,)
        