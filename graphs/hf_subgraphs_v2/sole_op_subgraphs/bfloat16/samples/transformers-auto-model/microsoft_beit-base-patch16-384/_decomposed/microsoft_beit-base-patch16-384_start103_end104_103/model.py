import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        in_0 += 23;  in_1 = in_0;  in_0 = None
        return (in_1,)
        