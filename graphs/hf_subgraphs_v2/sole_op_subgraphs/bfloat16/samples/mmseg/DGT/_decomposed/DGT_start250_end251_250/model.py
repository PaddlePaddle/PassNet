import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, in_0 : torch.Tensor):
        tmp_2 = torch.conv_transpose2d(in_0, w_1, w_0, (4, 4), (0, 0), (0, 0), 1, (1, 1));  in_0 = w_1 = w_0 = None
        return (tmp_2,)
        