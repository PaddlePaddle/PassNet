import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        unbind = in_0.unbind(0);  in_0 = None
        getitem = unbind[0]
        getitem_1 = unbind[1]
        getitem_2 = unbind[2];  unbind = None
        return (getitem, getitem_1, getitem_2)
        