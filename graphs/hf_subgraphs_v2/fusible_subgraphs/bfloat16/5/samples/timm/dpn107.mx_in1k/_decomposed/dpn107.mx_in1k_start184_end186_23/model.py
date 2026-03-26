import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        conv2d = torch.conv2d(in_1, in_0, None, (2, 2), (0, 0), (1, 1), 1);  in_1 = in_0 = None
        tmp_2 = conv2d[(slice(None, None, None), slice(None, 1024, None), slice(None, None, None), slice(None, None, None))]
        return (tmp_2, conv2d)
        