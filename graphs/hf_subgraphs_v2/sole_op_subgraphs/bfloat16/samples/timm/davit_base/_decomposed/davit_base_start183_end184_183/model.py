import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.pad(in_0, (0, 0, 0, 0), 'constant', None);  in_0 = None
        return (tmp_0,)
        