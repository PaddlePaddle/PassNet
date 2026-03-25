import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, in_0 : torch.Tensor):
        tmp_4 = torch.nn.functional.relu(in_0, inplace = True);  in_0 = None
        tmp_5 = torch.nn.functional.max_pool2d(tmp_4, 3, 2, 1, 1, ceil_mode = False, return_indices = False);  tmp_4 = None
        tmp_6 = torch.cat([tmp_5], 1)
        tmp_7 = torch.nn.functional.batch_norm(tmp_6, w_0, w_1, w_3, w_2, False, 0.1, 1e-05);  tmp_6 = w_0 = w_1 = w_3 = w_2 = None
        tmp_8 = torch.nn.functional.relu(tmp_7, inplace = True);  tmp_7 = None
        return (tmp_5, tmp_8)
        