import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.max_pool2d(in_0, 2, 2, 0, 1, ceil_mode = False, return_indices = False)
        tmp_1 = torch.nn.functional.max_pool2d(in_0, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_1 = None
        tmp_2 = torch.nn.functional.max_pool2d(in_0, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_2 = None
        tmp_3 = torch.nn.functional.max_pool2d(in_0, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  in_0 = None
        return (tmp_0, tmp_3)
        