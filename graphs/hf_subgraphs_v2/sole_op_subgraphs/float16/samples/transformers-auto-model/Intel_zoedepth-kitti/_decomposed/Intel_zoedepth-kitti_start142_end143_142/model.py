import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        sort = torch.sort(in_0, dim = 1);  in_0 = None
        getitem = sort[0]
        getitem_1 = sort[1];  sort = None
        return (getitem, getitem_1)
        