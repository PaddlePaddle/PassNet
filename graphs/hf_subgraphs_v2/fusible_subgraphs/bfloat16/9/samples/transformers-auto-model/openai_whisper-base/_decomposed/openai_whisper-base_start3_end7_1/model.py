import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_2 = in_1[in_2];  in_1 = in_2 = None
        tmp_3 = tmp_2.to(device(type='cuda', index=0));  tmp_2 = None
        tmp_4 = in_0 + tmp_3;  in_0 = tmp_3 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, p = 0.0, training = False);  tmp_4 = None
        return (tmp_5,)
        