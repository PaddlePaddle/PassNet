import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_1 = in_0[1]
        tmp_2 = in_0[0];  in_0 = None
        return (tmp_1, tmp_2)
        