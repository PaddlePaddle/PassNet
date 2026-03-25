import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        in_0 += in_1;  in_2 = in_0;  in_0 = in_1 = None
        return (in_2,)
        