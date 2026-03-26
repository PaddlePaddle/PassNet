import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_1 = torch.nn.functional.relu(in_1, inplace = True);  in_1 = None
        tmp_2 = torch.nn.functional.max_pool2d(tmp_1, 3, 1, 1, 1, ceil_mode = False, return_indices = False);  tmp_1 = None
        tmp_3 = torch.nn.functional.pad(tmp_2, [1, 1, 1, 1], 'reflect', None);  tmp_2 = None
        conv2d = torch.conv2d(tmp_3, in_0, stride = 2, groups = 64);  tmp_3 = in_0 = None
        return (conv2d,)
        