import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_1, w_0, (8, 8), (0, 0), (1, 1), 1);  in_0 = w_1 = w_0 = None
        return (conv2d,)
        