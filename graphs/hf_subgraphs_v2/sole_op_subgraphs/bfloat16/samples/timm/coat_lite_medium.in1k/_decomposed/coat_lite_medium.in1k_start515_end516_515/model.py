import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, in_0 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_1, w_0, (1, 1), (2, 2), (1, 1), 120);  in_0 = w_1 = w_0 = None
        return (conv2d,)
        