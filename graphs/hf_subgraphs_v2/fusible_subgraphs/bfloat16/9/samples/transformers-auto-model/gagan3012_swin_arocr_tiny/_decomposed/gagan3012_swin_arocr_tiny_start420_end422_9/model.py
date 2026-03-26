import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.normalize(in_0, dim = -1);  in_0 = None
        tmp_1 = tmp_0.transpose(-2, -1);  tmp_0 = None
        return (tmp_1,)
        