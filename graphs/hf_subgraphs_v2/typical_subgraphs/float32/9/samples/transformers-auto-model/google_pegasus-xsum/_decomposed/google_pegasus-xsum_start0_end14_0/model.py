import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = w_1
        tmp_3 = w_2
        tmp_4 = w_3
        tmp_5 = w_4
        tmp_6 = torch.arange(0, 10, device=device(type='cuda', index=0))
        tmp_7 = torch.full((10, 11), fill_value=-3.4028234663852886e+38, dtype=torch.float32, device=device(type='cuda', index=0))
        tmp_8 = torch.triu(tmp_7, diagonal=1)
        tmp_7 = None
        tmp_9 = torch.arange(11, device=device(type='cuda', index=0))
        tmp_10 = tmp_6.reshape(-1, 1)
        tmp_11 = tmp_9 > tmp_10
        tmp_9 = tmp_10 = None
        tmp_8 *= tmp_11
        tmp_12 = tmp_8
        tmp_8 = tmp_11 = None
        tmp_13 = tmp_12[None, None, slice(None, None, None), slice(None, None, None)]
        tmp_12 = None
        tmp_14 = tmp_13.expand(1, 1, -1, -1)
        tmp_13 = None
        tmp_15 = torch.nn.functional.embedding(tmp_6, tmp_1, None, None, 2.0, False, False)
        tmp_6 = tmp_1 = None
        tmp_16 = tmp_0 + tmp_15
        tmp_0 = tmp_15 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, p=0.1, training=False)
        tmp_16 = None
        tmp_18 = torch.nn.functional.layer_norm(tmp_17, (1024,), tmp_3, tmp_2, 1e-05)
        tmp_3 = tmp_2 = None
        tmp_19 = torch.nn.functional.linear(tmp_18, tmp_5, tmp_4)
        tmp_5 = tmp_4 = None
        return (tmp_14, tmp_17, tmp_18, tmp_19)