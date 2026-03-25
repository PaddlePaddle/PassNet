import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, in_0 : torch.Tensor):
        conv3d = torch.conv3d(in_0, w_0, None, (1, 2, 2), (1, 3, 3), (1, 1, 1), 1);  in_0 = w_0 = None
        return (conv3d,)
        