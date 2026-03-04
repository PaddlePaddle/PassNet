import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = torch.arange(512, device=device(type='cuda', index=0))
        tmp_2 = tmp_1.type_as(tmp_0)
        tmp_1 = None
        tmp_3 = torch.outer(tmp_2, tmp_0)
        tmp_2 = tmp_0 = None
        tmp_4 = torch.cat((tmp_3, tmp_3), dim=-1)
        tmp_3 = None
        tmp_5 = tmp_4.to(device(type='cuda', index=0))
        tmp_4 = None
        tmp_6 = tmp_5.cos()
        tmp_7 = tmp_6[None, None, slice(None, None, None), slice(None, None, None)]
        tmp_6 = None
        tmp_8 = tmp_5.sin()
        tmp_5 = None
        tmp_9 = tmp_8[None, None, slice(None, None, None), slice(None, None, None)]
        tmp_8 = None
        tmp_10 = tmp_7[slice(None, None, None), slice(None, None, None), slice(None, 512, None), slice(None, None, None)]
        tmp_11 = tmp_9[slice(None, None, None), slice(None, None, None), slice(None, 512, None), slice(None, None, None)]
        tmp_12 = in_1 * tmp_10
        tmp_10 = None
        tmp_13 = in_1.chunk(2, dim=-1)
        tmp_14 = tmp_13[0]
        tmp_15 = tmp_13[1]
        tmp_13 = None
        return (tmp_7, tmp_9, tmp_11, tmp_12, tmp_14, tmp_15)