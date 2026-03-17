import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = w_0
        tmp_3 = w_1
        tmp_4 = w_2
        tmp_5 = w_3
        tmp_6 = tmp_1.view(-1, 35)
        tmp_1 = None
        tmp_7 = torch.nn.functional.embedding(tmp_6, tmp_3, 57575, None, 2.0, False, False)
        tmp_6 = tmp_3 = None
        tmp_8 = tmp_7 * 32.0
        tmp_7 = None
        tmp_9 = torch.arange(0, 35, dtype=torch.int64, device=device(type='cuda'))
        tmp_10 = torch.nn.functional.embedding(tmp_9, tmp_2, None, None, 2.0, False, False)
        tmp_9 = tmp_2 = None
        tmp_11 = tmp_8 + tmp_10
        tmp_8 = tmp_10 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, p=0.1, training=False)
        tmp_11 = None
        tmp_13 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_14 = tmp_13.expand(1, 1, 35, 35)
        tmp_13 = None
        tmp_15 = tmp_14.to(torch.float32)
        tmp_14 = None
        tmp_16 = torch.tensor(1.0, dtype=torch.float32)
        tmp_17 = tmp_16 - tmp_15
        tmp_16 = tmp_15 = None
        tmp_18 = tmp_17.to(torch.bool)
        tmp_19 = tmp_17.masked_fill(tmp_18, -3.4028234663852886e+38)
        tmp_17 = tmp_18 = None
        tmp_20 = torch.nn.functional.linear(tmp_12, tmp_5, tmp_4)
        tmp_5 = tmp_4 = None
        return (tmp_19, tmp_12, tmp_20)