import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5, w_6):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = w_1
        tmp_3 = w_2
        tmp_4 = w_3
        tmp_5 = w_4
        tmp_6 = w_5
        tmp_7 = w_6
        tmp_8 = tmp_2[slice(None, None, None), slice(None, 512, None)]
        tmp_2 = None
        tmp_9 = tmp_8.expand(1, 512)
        tmp_8 = None
        tmp_10 = tmp_1[slice(None, None, None), slice(0, 512, None)]
        tmp_1 = None
        tmp_11 = torch.nn.functional.embedding(tmp_0, tmp_7, 0, None, 2.0, False, False)
        tmp_0 = tmp_7 = None
        tmp_12 = torch.nn.functional.embedding(tmp_9, tmp_6, None, None, 2.0, False, False)
        tmp_9 = tmp_6 = None
        tmp_13 = tmp_11 + tmp_12
        tmp_11 = tmp_12 = None
        tmp_14 = torch.nn.functional.embedding(tmp_10, tmp_5, None, None, 2.0, False, False)
        tmp_10 = tmp_5 = None
        tmp_13 += tmp_14
        tmp_15 = tmp_13
        tmp_13 = tmp_14 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (384,), tmp_4, tmp_3, 1e-12)
        tmp_15 = tmp_4 = tmp_3 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, 0.1, False, False)
        tmp_16 = None
        tmp_18 = torch.ones((1, 512), device=device(type='cuda', index=0))
        tmp_19 = tmp_18[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_18 = None
        tmp_20 = tmp_19.expand(1, 1, 512, 512)
        tmp_19 = None
        tmp_21 = tmp_20.to(torch.float32)
        tmp_20 = None
        tmp_22 = 1.0 - tmp_21
        tmp_21 = None
        tmp_23 = tmp_22.to(torch.bool)
        tmp_24 = tmp_22.masked_fill(tmp_23, -3.4028234663852886e+38)
        tmp_22 = tmp_23 = None
        return (tmp_17, tmp_24)