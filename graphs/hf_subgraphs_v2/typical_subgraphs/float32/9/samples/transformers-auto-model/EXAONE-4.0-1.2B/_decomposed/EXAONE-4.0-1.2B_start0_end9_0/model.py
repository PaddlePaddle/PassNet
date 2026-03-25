import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = w_0
        tmp_3 = torch.nn.functional.embedding(tmp_1, tmp_2, 0, None, 2.0, False, False)
        tmp_1 = tmp_2 = None
        tmp_4 = torch.arange(0, 19, device=device(type='cuda'))
        tmp_5 = tmp_4.unsqueeze(0)
        tmp_6 = tmp_0.to(device=device(type='cuda'), dtype=torch.bool)
        tmp_0 = None
        tmp_7 = torch.arange(19, device=device(type='cuda'))
        tmp_7 += 0
        tmp_8 = tmp_7
        tmp_7 = None
        tmp_9 = torch.arange(1, device=device(type='cuda'))
        tmp_10 = torch.arange(1, device=device(type='cuda'))
        tmp_11 = torch._functorch.vmap.lazy_load_decompositions()
        tmp_11 = None
        return (tmp_3, tmp_4, tmp_5, tmp_6, tmp_8, tmp_9, tmp_10)