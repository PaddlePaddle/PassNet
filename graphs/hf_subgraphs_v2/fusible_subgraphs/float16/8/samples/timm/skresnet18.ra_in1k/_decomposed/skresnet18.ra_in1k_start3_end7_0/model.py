import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.max_pool2d(in_0, 3, 2, 1, 1, ceil_mode = False, return_indices = False);  in_0 = None
        split = torch.functional.split(tmp_0, 32, 1)
        tmp_2 = split[0]
        tmp_3 = split[1];  split = None
        return (tmp_2, tmp_3, tmp_0)
        