import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.repeat_interleave(in_0, 2, 3);  in_0 = None
        return (tmp_0,)
        