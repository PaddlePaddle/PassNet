import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        tmp_2 = torch.nn.functional.relu(in_2, inplace = False);  in_2 = None
        tmp_3 = in_1 * tmp_2;  in_1 = tmp_2 = None
        tmp_4 = tmp_3 + in_0;  tmp_3 = in_0 = None
        tmp_5 = torch.nn.functional.max_pool2d(in_3, 2, 1, 0, 1, ceil_mode = True, return_indices = False);  in_3 = None
        tmp_6 = torch.cat([tmp_5, tmp_4], dim = 1);  tmp_5 = tmp_4 = None
        return (tmp_6,)
        