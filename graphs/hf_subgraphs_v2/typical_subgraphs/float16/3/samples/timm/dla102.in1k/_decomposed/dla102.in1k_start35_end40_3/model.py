import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        in_1 += in_0;  in_2 = in_1;  in_1 = in_0 = None
        tmp_1 = torch.nn.functional.relu(in_2, inplace = True);  in_2 = None
        tmp_2 = torch.nn.functional.max_pool2d(tmp_1, 2, 2, 0, 1, ceil_mode = False, return_indices = False)
        tmp_3 = torch.nn.functional.max_pool2d(tmp_1, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_3 = None
        tmp_4 = torch.nn.functional.max_pool2d(tmp_1, 2, 2, 0, 1, ceil_mode = False, return_indices = False)
        return (tmp_2, tmp_4, tmp_1)
        