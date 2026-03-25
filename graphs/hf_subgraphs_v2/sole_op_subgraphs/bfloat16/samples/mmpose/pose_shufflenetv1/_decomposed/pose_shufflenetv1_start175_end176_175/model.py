import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = torch.cat((in_1, in_0), 1);  in_1 = in_0 = None
        return (tmp_0,)
        