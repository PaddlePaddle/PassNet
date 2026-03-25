import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.relu(in_0, inplace = True);  in_0 = None
        tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 3, 1, 1, 1, ceil_mode = False, return_indices = False);  tmp_0 = None
        return (tmp_1,)
        