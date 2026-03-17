import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3, w_4, w_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = w_0
        tmp_3 = w_1
        tmp_4 = w_2
        tmp_5 = w_3
        tmp_6 = w_4
        tmp_7 = w_5
        tmp_8 = tmp_1.view(-1, 10)
        tmp_1 = None
        tmp_9 = torch.nn.functional.embedding(tmp_8, tmp_3, 0, None, 2.0, False, False)
        tmp_8 = tmp_3 = None
        tmp_10 = tmp_9 * 32.0
        tmp_9 = None
        tmp_11 = torch.arange(0, 10, dtype=torch.int64, device=device(type='cuda', index=0))
        tmp_12 = torch.nn.functional.embedding(tmp_11, tmp_2, None, None, 2.0, False, False)
        tmp_11 = tmp_2 = None
        tmp_13 = tmp_10 + tmp_12
        tmp_10 = tmp_12 = None
        tmp_14 = torch.nn.functional.dropout(tmp_13, p=0.1, training=False)
        tmp_13 = None
        tmp_15 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_16 = tmp_15.expand(1, 1, 10, 10)
        tmp_15 = None
        tmp_17 = tmp_16.to(torch.bfloat16)
        tmp_16 = None
        tmp_18 = torch.tensor(1.0, dtype=torch.bfloat16)
        tmp_19 = tmp_18 - tmp_17
        tmp_18 = tmp_17 = None
        tmp_20 = tmp_19.to(torch.bool)
        tmp_21 = tmp_19.masked_fill(tmp_20, -3.3895313892515355e+38)
        tmp_19 = tmp_20 = None
        tmp_22 = torch.nn.functional.layer_norm(tmp_14, (1024,), tmp_5, tmp_4, 1e-05)
        tmp_5 = tmp_4 = None
        tmp_23 = torch.nn.functional.linear(tmp_22, tmp_7, tmp_6)
        tmp_7 = tmp_6 = None
        return (tmp_21, tmp_14, tmp_22, tmp_23)