import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        tmp_0 = torch.cat([in_1, in_2, in_3, in_4, in_0], dim = 1);  in_1 = in_2 = in_3 = in_4 = in_0 = None
        return (tmp_0,)
        