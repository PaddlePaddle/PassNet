import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (1, 1), (1, 1), 1);  in_2 = in_1 = in_0 = None
        tmp_3 = torch.nn.functional.interpolate(conv2d, size = (640, 640), mode = 'bilinear');  conv2d = None
        return (tmp_3,)
        