import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        tmp_2 = torch.nn.functional.interpolate(in_2, None, 8.0, 'nearest', None, recompute_scale_factor = None);  in_2 = None
        in_3 += tmp_2;  in_4 = in_3;  in_3 = tmp_2 = None
        tmp_4 = torch.nn.functional.relu(in_4, inplace = True);  in_4 = None
        conv2d = torch.conv2d(tmp_4, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_4 = in_1 = in_0 = None
        return (conv2d,)
        