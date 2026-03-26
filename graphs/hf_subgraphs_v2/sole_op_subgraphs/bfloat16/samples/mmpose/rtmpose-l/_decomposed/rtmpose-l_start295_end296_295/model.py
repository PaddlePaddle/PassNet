import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        bmm = torch.bmm(in_0, in_1);  in_0 = in_1 = None
        return (bmm,)
        