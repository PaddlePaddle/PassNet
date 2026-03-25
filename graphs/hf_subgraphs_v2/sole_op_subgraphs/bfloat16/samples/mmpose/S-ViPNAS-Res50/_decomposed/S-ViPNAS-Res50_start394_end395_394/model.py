import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, in_0 : torch.Tensor):
        tmp_1 = torch.conv_transpose2d(in_0, w_0, None, (2, 2), (1, 1), (0, 0), 16, (1, 1));  in_0 = w_0 = None
        return (tmp_1,)
        