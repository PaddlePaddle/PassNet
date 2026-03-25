import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        chunk = in_0.chunk(3, dim = -1);  in_0 = None
        getitem = chunk[0]
        getitem_1 = chunk[1]
        getitem_2 = chunk[2];  chunk = None
        return (getitem, getitem_1, getitem_2)
        