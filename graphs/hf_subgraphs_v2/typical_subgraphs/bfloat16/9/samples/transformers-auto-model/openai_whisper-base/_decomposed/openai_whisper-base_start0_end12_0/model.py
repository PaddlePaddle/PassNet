import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor):
        tmp_2 = torch.arange(0, 1, device = device(type='cuda', index=0))
        tmp_3 = tmp_2.unsqueeze(0)
        tmp_4 = tmp_3.repeat(1, 1);  tmp_3 = None
        tmp_5 = w_0[tmp_4];  w_0 = tmp_4 = None
        tmp_6 = tmp_5.to(device(type='cuda', index=0));  tmp_5 = None
        tmp_7 = in_0 + tmp_6;  in_0 = tmp_6 = None
        tmp_8 = torch.nn.functional.dropout(tmp_7, p = 0.0, training = False);  tmp_7 = None
        tmp_9 = torch.arange(1, device = device(type='cuda', index=0))
        tmp_9 += 0;  tmp_10 = tmp_9;  tmp_9 = None
        tmp_11 = torch.arange(1, device = device(type='cuda', index=0))
        tmp_12 = torch.arange(1, device = device(type='cuda', index=0))
        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions = None
        return (tmp_2, tmp_8, tmp_10, tmp_11, tmp_12)
        