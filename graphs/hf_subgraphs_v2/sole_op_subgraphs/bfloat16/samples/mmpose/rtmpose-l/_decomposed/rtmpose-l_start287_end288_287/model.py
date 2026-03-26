import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        unbind = torch.unbind(in_0, dim = 2);  in_0 = None
        getitem = unbind[0]
        getitem_1 = unbind[1];  unbind = None
        return (getitem, getitem_1)
        