import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_1 = in_0.to(device = device(type='cuda', index=0), dtype = torch.bool);  in_0 = None
        tmp_2 = torch.arange(2, device = device(type='cuda', index=0))
        tmp_2 += 0;  tmp_3 = tmp_2;  tmp_2 = None
        tmp_4 = tmp_1[(slice(None, None, None), tmp_3)];  tmp_1 = tmp_3 = None
        tmp_5 = torch.arange(2, device = device(type='cuda', index=0))
        tmp_5 += 0;  tmp_6 = tmp_5;  tmp_5 = None
        tmp_7 = in_1.view(-1, 1);  in_1 = None
        tmp_8 = tmp_6 <= tmp_7;  tmp_6 = tmp_7 = None
        tmp_9 = tmp_8[(None, None, slice(None, None, None), slice(None, None, None))];  tmp_8 = None
        tmp_10 = tmp_9.expand(1, -1, -1, -1);  tmp_9 = None
        tmp_11 = tmp_4[(slice(None, None, None), None, None, slice(None, None, None))];  tmp_4 = None
        tmp_12 = tmp_10 * tmp_11;  tmp_10 = tmp_11 = None
        return (tmp_12,)
        