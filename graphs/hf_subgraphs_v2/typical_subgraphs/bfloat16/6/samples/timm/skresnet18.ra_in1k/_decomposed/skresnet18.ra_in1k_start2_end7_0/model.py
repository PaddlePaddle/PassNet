import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.relu(in_0, inplace = True);  in_0 = None
        tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 3, 2, 1, 1, ceil_mode = False, return_indices = False);  tmp_0 = None
        split = torch.functional.split(tmp_1, 32, 1)
        tmp_3 = split[0]
        tmp_4 = split[1];  split = None
        return (tmp_3, tmp_4, tmp_1)
        