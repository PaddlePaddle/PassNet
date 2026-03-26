import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_0 = torch.where(in_2, in_1, in_0);  in_2 = in_1 = in_0 = None
        return (tmp_0,)
        