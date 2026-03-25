import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tensor_split = in_0.tensor_split(2, -1);  in_0 = None
        getitem = tensor_split[0]
        getitem_1 = tensor_split[1];  tensor_split = None
        return (getitem, getitem_1)
        