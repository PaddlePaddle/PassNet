import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_1 = torch.zeros((1, 256, 4), dtype = torch.int64, device = device(type='cuda', index=0))
        tmp_2 = in_0.unsqueeze(1);  in_0 = None
        tmp_3 = tmp_2.unsqueeze(2);  tmp_2 = None
        return (tmp_3, tmp_1)
        