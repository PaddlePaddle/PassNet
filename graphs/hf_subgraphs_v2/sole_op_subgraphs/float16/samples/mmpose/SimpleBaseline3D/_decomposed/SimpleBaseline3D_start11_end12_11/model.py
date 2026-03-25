import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.dropout(in_0, 0.5, False, False);  in_0 = None
        return (tmp_0,)
        