import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 1);  in_1 = in_0 = None
        tmp_2 = torch.nn.functional.max_pool2d(conv2d, 3, 2, 1, 1, ceil_mode = False, return_indices = False);  conv2d = None
        return (tmp_2,)
        