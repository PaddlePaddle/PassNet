import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        meshgrid = torch.functional.meshgrid(in_1, in_0, indexing = 'xy');  in_1 = in_0 = None
        getitem = meshgrid[0]
        getitem_1 = meshgrid[1];  meshgrid = None
        return (getitem, getitem_1)
        