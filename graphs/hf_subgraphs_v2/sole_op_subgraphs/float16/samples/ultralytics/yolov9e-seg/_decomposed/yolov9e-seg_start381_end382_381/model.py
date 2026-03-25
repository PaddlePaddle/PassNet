import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        split = in_0.split([64, 128, 256, 512, 1024], dim = 1);  in_0 = None
        getitem = split[0]
        getitem_1 = split[1]
        getitem_2 = split[2]
        getitem_3 = split[3]
        getitem_4 = split[4];  split = None
        return (getitem, getitem_1, getitem_2, getitem_3, getitem_4)
        