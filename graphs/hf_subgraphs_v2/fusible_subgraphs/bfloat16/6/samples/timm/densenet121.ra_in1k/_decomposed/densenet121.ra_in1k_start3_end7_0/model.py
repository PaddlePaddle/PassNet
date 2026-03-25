import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        tmp_4 = torch.nn.functional.max_pool2d(in_4, 3, 2, 1, 1, ceil_mode = False, return_indices = False);  in_4 = None
        tmp_5 = torch.cat([tmp_4], 1)
        tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  tmp_5 = in_0 = in_1 = in_3 = in_2 = None
        tmp_7 = torch.nn.functional.relu(tmp_6, inplace = True);  tmp_6 = None
        return (tmp_4, tmp_7)
        