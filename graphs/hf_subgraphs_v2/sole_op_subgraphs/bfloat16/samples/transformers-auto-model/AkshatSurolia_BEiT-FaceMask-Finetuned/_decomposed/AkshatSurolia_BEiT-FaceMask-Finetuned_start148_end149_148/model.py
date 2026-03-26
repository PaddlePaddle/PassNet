import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        meshgrid = torch.functional.meshgrid(in_0, in_1, indexing = 'ij');  in_0 = in_1 = None
        getitem = meshgrid[0]
        getitem_1 = meshgrid[1];  meshgrid = None
        return (getitem, getitem_1)
        