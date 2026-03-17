import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = w_0
        tmp_3 = w_1
        tmp_4 = torch.nn.functional.embedding(tmp_1, tmp_3, 0, None, 2.0, False, False)
        tmp_1 = tmp_3 = None
        tmp_5 = tmp_2.to(torch.float16)
        tmp_2 = None
        tmp_6 = tmp_4 * tmp_5
        tmp_4 = tmp_5 = None
        tmp_7 = torch.arange(0, 20, device=device(type='cuda'))
        tmp_8 = tmp_7.unsqueeze(0)
        tmp_9 = tmp_0.to(device=device(type='cuda'), dtype=torch.bool)
        tmp_0 = None
        tmp_10 = torch.arange(20, device=device(type='cuda'))
        tmp_10 += 0
        tmp_11 = tmp_10
        tmp_10 = None
        tmp_12 = torch.arange(1, device=device(type='cuda'))
        tmp_13 = torch.arange(1, device=device(type='cuda'))
        tmp_14 = torch._functorch.vmap.lazy_load_decompositions()
        tmp_14 = None
        return (tmp_6, tmp_7, tmp_8, tmp_9, tmp_11, tmp_12, tmp_13)