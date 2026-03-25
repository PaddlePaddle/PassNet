import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        conv2d = torch.conv2d(in_0, in_1, None, (1, 1), (0, 0), (1, 1), 1);  in_0 = in_1 = None
        return (conv2d,)
        