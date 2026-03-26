import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.softmax(in_0, dim = 1);  in_0 = None
        tmp_1 = torch.linspace(0, 4, steps = 5, device = device(type='cuda', index=0))
        tmp_2 = tmp_0 * tmp_1;  tmp_0 = tmp_1 = None
        tmp_3 = tmp_2.sum(dim = 1);  tmp_2 = None
        tmp_4 = 5 - tmp_3;  tmp_3 = None
        return (tmp_4,)
        