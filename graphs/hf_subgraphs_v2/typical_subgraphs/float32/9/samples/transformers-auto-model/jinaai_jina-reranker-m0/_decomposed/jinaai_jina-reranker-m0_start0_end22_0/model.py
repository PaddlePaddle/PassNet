import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = w_0
        tmp_3 = torch.nn.functional.embedding(tmp_1, tmp_2, None, None, 2.0, False, False)
        tmp_1 = tmp_2 = None
        tmp_4 = tmp_0.long()
        tmp_5 = tmp_4.cumsum(-1)
        tmp_4 = None
        tmp_6 = tmp_5 - 1
        tmp_5 = None
        tmp_7 = tmp_0.__eq__(0)
        tmp_8 = tmp_6.masked_fill_(tmp_7, 1)
        tmp_7 = tmp_8 = None
        tmp_9 = tmp_6.unsqueeze(0)
        tmp_6 = None
        tmp_10 = tmp_9.expand(3, -1, -1)
        tmp_9 = None
        tmp_11 = tmp_10.to(device(type='cuda', index=0))
        tmp_10 = None
        tmp_12 = tmp_11.max(0, keepdim=False)
        tmp_13 = tmp_12[0]
        tmp_12 = None
        tmp_14 = tmp_13.max(-1, keepdim=True)
        tmp_13 = None
        tmp_15 = tmp_14[0]
        tmp_14 = None
        tmp_16 = tmp_15 + 1
        tmp_15 = None
        tmp_17 = tmp_16 - 9
        tmp_16 = None
        tmp_18 = torch.arange(0, 9, device=device(type='cuda', index=0))
        tmp_19 = tmp_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
        tmp_0 = None
        tmp_20 = torch.arange(9, device=device(type='cuda', index=0))
        tmp_20 += 0
        tmp_21 = tmp_20
        tmp_20 = None
        tmp_22 = torch.arange(1, device=device(type='cuda', index=0))
        tmp_23 = torch.arange(1, device=device(type='cuda', index=0))
        tmp_24 = torch._functorch.vmap.lazy_load_decompositions()
        tmp_24 = None
        return (tmp_3, tmp_11, tmp_17, tmp_18, tmp_19, tmp_21, tmp_22, tmp_23)