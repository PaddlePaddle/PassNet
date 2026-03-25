import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_2 = in_0.view(-1, 64);  in_0 = None
        tmp_3 = in_1[(slice(None, None, None), slice(None, 64, None))];  in_1 = None
        return (tmp_2, tmp_3)
        