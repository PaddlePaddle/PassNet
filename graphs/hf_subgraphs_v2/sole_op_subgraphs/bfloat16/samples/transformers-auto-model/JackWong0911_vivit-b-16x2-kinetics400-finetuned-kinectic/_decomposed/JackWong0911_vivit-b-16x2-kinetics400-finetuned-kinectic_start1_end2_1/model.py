import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, in_0 : torch.Tensor):
        conv3d = torch.conv3d(in_0, w_1, w_0, (2, 16, 16), (0, 0, 0), (1, 1, 1), 1);  in_0 = w_1 = w_0 = None
        return (conv3d,)
        