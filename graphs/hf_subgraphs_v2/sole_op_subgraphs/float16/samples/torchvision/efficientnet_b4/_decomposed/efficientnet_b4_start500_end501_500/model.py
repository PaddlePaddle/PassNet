import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.dropout(in_0, 0.4, False, True);  in_0 = None
        return (tmp_0,)
        