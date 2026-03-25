import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.leaky_relu(in_0, 0.01, True);  in_0 = None
        split = tmp_0.split(64, dim = 1);  tmp_0 = None
        tmp_2 = split[0]
        tmp_3 = split[1];  split = None
        return (tmp_3, tmp_2)
        