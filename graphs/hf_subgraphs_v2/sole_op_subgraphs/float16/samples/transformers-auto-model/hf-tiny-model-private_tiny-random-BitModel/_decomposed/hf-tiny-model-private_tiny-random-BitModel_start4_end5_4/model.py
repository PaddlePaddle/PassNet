import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        conv2d = torch.conv2d(in_0, in_1, None, (2, 2), (3, 3), (1, 1), 1);  in_0 = in_1 = None
        return (conv2d,)
        