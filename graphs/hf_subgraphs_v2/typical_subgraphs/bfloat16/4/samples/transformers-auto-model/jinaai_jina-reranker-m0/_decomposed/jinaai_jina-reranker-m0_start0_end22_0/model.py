import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_3 = torch.nn.functional.embedding(in_1, in_2, None, None, 2.0, False, False);  in_1 = in_2 = None
        tmp_4 = in_0.long()
        tmp_5 = tmp_4.cumsum(-1);  tmp_4 = None
        tmp_6 = tmp_5 - 1;  tmp_5 = None
        tmp_7 = in_0.__eq__(0)
        tmp_8 = tmp_6.masked_fill_(tmp_7, 1);  tmp_7 = tmp_8 = None
        tmp_9 = tmp_6.unsqueeze(0);  tmp_6 = None
        tmp_10 = tmp_9.expand(3, -1, -1);  tmp_9 = None
        tmp_11 = tmp_10.to(device(type='cuda', index=0));  tmp_10 = None
        max_1 = tmp_11.max(0, keepdim = False)
        tmp_13 = max_1[0];  max_1 = None
        max_2 = tmp_13.max(-1, keepdim = True);  tmp_13 = None
        tmp_15 = max_2[0];  max_2 = None
        tmp_16 = tmp_15 + 1;  tmp_15 = None
        tmp_17 = tmp_16 - 9;  tmp_16 = None
        tmp_18 = torch.arange(0, 256, device = device(type='cuda', index=0))
        tmp_19 = in_0.to(device = device(type='cuda', index=0), dtype = torch.bool);  in_0 = None
        tmp_20 = torch.arange(256, device = device(type='cuda', index=0))
        tmp_20 += 0;  tmp_21 = tmp_20;  tmp_20 = None
        tmp_22 = torch.arange(1, device = device(type='cuda', index=0))
        tmp_23 = torch.arange(1, device = device(type='cuda', index=0))
        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions = None
        return (tmp_3, tmp_11, tmp_17, tmp_18, tmp_19, tmp_21, tmp_22, tmp_23)
        