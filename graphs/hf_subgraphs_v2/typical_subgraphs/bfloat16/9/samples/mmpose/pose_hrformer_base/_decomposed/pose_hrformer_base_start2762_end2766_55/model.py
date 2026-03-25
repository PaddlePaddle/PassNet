import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_2 = torch.nn.functional.interpolate(in_0, None, 8.0, 'bilinear', False, recompute_scale_factor = None);  in_0 = None
        in_1 += tmp_2;  in_2 = in_1;  in_1 = tmp_2 = None
        tmp_4 = torch.nn.functional.relu(in_2, inplace = True);  in_2 = None
        conv2d = torch.conv2d(tmp_4, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_4 = w_1 = w_0 = None
        return (conv2d,)
        