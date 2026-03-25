import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        split = in_0.split([64], dim = 1);  in_0 = None
        getitem = split[0];  split = None
        return (getitem,)
        