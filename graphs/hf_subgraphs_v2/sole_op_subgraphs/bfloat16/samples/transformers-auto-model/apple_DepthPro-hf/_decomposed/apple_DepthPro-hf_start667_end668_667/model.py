import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        split_with_sizes = torch.split_with_sizes(in_0, [25, 9, 1]);  in_0 = None
        getitem = split_with_sizes[0]
        getitem_1 = split_with_sizes[1]
        getitem_2 = split_with_sizes[2];  split_with_sizes = None
        return (getitem, getitem_1, getitem_2)
        