import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.sigmoid(in_0);  in_0 = None
        split = torch.functional.split(tmp_0, [20, 40], dim = 1);  tmp_0 = None
        tmp_2 = split[0]
        tmp_3 = split[1];  split = None
        return (tmp_2, tmp_3)
        