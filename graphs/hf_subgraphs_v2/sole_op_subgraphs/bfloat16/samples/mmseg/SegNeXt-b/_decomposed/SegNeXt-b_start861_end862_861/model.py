import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        bmm = in_0.bmm(in_1);  in_0 = in_1 = None
        return (bmm,)
        