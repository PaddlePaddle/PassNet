import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_1 = in_0.repeat(1, 1, 1);  in_0 = None
        tmp_2 = tmp_1.transpose(1, 0);  tmp_1 = None
        tmp_3 = in_1.transpose(1, 0);  in_1 = None
        return (tmp_3, tmp_2)
        