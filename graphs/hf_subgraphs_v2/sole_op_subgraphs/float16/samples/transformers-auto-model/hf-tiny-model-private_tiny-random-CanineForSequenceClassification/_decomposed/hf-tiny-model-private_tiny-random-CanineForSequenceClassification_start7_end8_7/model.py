import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.max_pool1d(in_0, 4, 4, 0, 1, ceil_mode = False, return_indices = False);  in_0 = None
        return (tmp_0,)
        