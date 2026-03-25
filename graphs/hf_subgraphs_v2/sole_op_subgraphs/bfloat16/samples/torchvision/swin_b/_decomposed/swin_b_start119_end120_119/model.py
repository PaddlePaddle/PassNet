import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_0 = torch.cat([in_0, in_1, in_2, in_3], -1);  in_0 = in_1 = in_2 = in_3 = None
        return (tmp_0,)
        