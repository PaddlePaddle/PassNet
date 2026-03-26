import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, in_0, in_1):
        conv2d = torch.conv2d(in_0, in_1, w_0, (1, 1), (1, 1), (1, 1), 1);  in_0 = in_1 = w_0 = None
        return (conv2d,)
        