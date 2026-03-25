import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_2 = torch.nn.functional.relu(in_0, inplace = False);  in_0 = None
        tmp_3 = w_1 * tmp_2;  w_1 = tmp_2 = None
        tmp_4 = tmp_3 + w_0;  tmp_3 = w_0 = None
        tmp_5 = torch.nn.functional.max_pool2d(in_1, 2, 1, 0, 1, ceil_mode = True, return_indices = False);  in_1 = None
        tmp_6 = torch.cat([tmp_5, tmp_4], dim = 1);  tmp_5 = tmp_4 = None
        return (tmp_6,)
        