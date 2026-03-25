import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        conv2d = torch.conv2d(in_0, in_1, padding = 0);  in_0 = in_1 = None
        return (conv2d,)
        