import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        in_1 += in_0;  in_2 = in_1;  in_1 = in_0 = None
        return (in_2,)
        