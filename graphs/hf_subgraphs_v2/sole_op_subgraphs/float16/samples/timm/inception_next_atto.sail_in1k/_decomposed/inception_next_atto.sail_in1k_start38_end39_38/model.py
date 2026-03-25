import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        split = torch.functional.split(in_0, (20, 20, 20, 20), dim = 1);  in_0 = None
        getitem = split[0]
        getitem_1 = split[1]
        getitem_2 = split[2]
        getitem_3 = split[3];  split = None
        return (getitem, getitem_1, getitem_2, getitem_3)
        