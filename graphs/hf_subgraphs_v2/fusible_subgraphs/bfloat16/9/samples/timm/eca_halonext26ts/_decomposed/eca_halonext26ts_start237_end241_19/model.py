import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        split = torch.functional.split(in_0, [16, 64], dim = -1);  in_0 = None
        tmp_1 = split[0]
        tmp_2 = split[1];  split = None
        tmp_3 = tmp_1.transpose(-1, -2);  tmp_1 = None
        return (tmp_3, tmp_2)
        