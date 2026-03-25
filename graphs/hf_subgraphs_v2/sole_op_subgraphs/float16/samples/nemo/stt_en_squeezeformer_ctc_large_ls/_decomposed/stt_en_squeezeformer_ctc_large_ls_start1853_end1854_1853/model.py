import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.repeat_interleave(in_0, repeats = 2, dim = 1);  in_0 = None
        return (tmp_0,)
        