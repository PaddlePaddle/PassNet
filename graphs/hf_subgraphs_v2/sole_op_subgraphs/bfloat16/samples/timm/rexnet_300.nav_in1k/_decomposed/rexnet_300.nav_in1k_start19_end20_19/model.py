import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, in_0 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_0, None, (1, 1), (1, 1), (1, 1), 492);  in_0 = w_0 = None
        return (conv2d,)
        