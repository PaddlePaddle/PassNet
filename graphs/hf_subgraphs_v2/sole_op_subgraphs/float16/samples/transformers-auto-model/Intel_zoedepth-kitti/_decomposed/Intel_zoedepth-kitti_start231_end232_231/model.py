import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.clip(in_0, 0.001, 10.0);  in_0 = None
        return (tmp_0,)
        