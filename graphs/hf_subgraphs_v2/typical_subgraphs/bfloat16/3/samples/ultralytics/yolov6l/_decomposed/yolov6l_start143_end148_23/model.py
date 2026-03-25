import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.relu(in_0, inplace = True);  in_0 = None
        tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode = False, return_indices = False)
        tmp_2 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode = False, return_indices = False)
        tmp_3 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode = False, return_indices = False)
        tmp_4 = torch.cat([tmp_0, tmp_1, tmp_2, tmp_3], 1);  tmp_0 = tmp_1 = tmp_2 = tmp_3 = None
        return (tmp_4,)
        