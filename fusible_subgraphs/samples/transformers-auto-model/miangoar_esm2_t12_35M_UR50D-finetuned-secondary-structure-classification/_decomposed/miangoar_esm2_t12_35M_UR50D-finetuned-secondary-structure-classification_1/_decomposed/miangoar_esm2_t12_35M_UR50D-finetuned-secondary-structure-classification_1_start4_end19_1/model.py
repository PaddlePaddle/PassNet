import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0):
        tmp_0 = torch.arange(13, device=device(type='cuda', index=0))
        tmp_1 = tmp_0.type_as(w_0)
        tmp_0 = None
        tmp_2 = torch.outer(tmp_1, w_0)
        tmp_1 = None
        tmp_3 = torch.cat((tmp_2, tmp_2), dim=-1)
        tmp_2 = None
        tmp_4 = tmp_3.to(device(type='cuda', index=0))
        tmp_3 = None
        tmp_5 = tmp_4.cos()
        tmp_6 = tmp_5[None, None, slice(None, None, None), slice(None, None, None)]
        tmp_5 = None
        tmp_7 = tmp_4.sin()
        tmp_4 = None
        tmp_8 = tmp_7[None, None, slice(None, None, None), slice(None, None, None)]
        tmp_7 = None
        tmp_9 = tmp_6[slice(None, None, None), slice(None, None, None), slice(None, 13, None), slice(None, None, None)]
        tmp_10 = tmp_8[slice(None, None, None), slice(None, None, None), slice(None, 13, None), slice(None, None, None)]
        tmp_11 = in_0 * tmp_9
        tmp_9 = None
        tmp_12 = in_0.chunk(2, dim=-1)
        tmp_13 = tmp_12[0]
        tmp_14 = tmp_12[1]
        tmp_12 = None
        return (tmp_6, tmp_8, tmp_10, tmp_11, tmp_13, tmp_14)