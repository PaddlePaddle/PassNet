import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        split = torch.functional.split(in_0, [2048, 2048, 4], 1);  in_0 = None
        getitem = split[0]
        getitem_1 = split[1]
        getitem_2 = split[2];  split = None
        return (getitem, getitem_1, getitem_2)
        