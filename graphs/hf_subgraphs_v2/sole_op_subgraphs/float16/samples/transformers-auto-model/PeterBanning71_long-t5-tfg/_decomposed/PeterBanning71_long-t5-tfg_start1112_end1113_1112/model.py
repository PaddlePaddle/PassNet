import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_1 = torch.nn.functional.pad(in_0, (0, 116, 0, 0), 'constant', 0);  in_0 = None
        return (tmp_1,)
        