import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        max_1 = torch.max(in_0, -1, keepdim = True);  in_0 = None
        getitem = max_1[0]
        getitem_1 = max_1[1];  max_1 = None
        return (getitem, getitem_1)
        