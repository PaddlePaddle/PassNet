import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        split = torch.functional.split(in_0, [80, 240], dim = 1);  in_0 = None
        tmp_1 = split[0]
        tmp_2 = split[1];  split = None
        return (tmp_1, tmp_2)
        