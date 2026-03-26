import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        split = in_0.split(384, dim = 1);  in_0 = None
        getitem = split[0]
        getitem_1 = split[1];  split = None
        return (getitem, getitem_1)
        