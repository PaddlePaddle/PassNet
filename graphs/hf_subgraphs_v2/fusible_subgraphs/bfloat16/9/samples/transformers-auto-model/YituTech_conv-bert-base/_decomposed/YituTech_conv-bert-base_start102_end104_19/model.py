import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = torch.cat([in_0, in_1], 2);  in_0 = in_1 = None
        tmp_1 = tmp_0.view(1, 11, 768);  tmp_0 = None
        return (tmp_1,)
        