import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.hardswish(in_0, True);  in_0 = None
        return (tmp_0,)
        