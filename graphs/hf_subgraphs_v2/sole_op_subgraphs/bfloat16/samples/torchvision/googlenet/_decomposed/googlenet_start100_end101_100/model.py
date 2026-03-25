import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.max_pool2d(in_0, 3, 1, 1, 1, ceil_mode = True, return_indices = False);  in_0 = None
        return (tmp_0,)
        