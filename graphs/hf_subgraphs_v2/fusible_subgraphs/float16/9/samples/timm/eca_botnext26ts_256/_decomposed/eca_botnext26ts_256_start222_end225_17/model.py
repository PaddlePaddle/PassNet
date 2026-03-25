import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_1 = in_1 * 0.25;  in_1 = None
        tmp_2 = in_2.reshape(4, 8, 8, -1);  in_2 = None
        tmp_3 = in_0.transpose(-1, -2);  in_0 = None
        return (tmp_1, tmp_2, tmp_3)
        