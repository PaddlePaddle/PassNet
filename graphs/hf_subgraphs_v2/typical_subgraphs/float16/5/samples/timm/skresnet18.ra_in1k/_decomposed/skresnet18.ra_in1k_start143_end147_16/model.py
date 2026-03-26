import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = torch.nn.functional.relu(in_1, inplace = True);  in_1 = None
        tmp_1 = torch.stack([in_0, tmp_0], dim = 1);  in_0 = tmp_0 = None
        tmp_2 = tmp_1.sum(1)
        tmp_3 = tmp_2.mean((2, 3), keepdim = True);  tmp_2 = None
        return (tmp_1, tmp_3)
        