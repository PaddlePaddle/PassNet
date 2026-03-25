import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = tmp_0.__eq__(1)
        tmp_6 = tmp_5.to(torch.float32)
        tmp_5 = None
        tmp_6 *= -3.4028234663852886e+38
        tmp_7 = tmp_6
        tmp_6 = None
        tmp_8 = tmp_7.unsqueeze(1)
        tmp_7 = None
        tmp_9 = tmp_8.unsqueeze(1)
        tmp_8 = None
        tmp_10 = torch.nn.functional.embedding(tmp_0, tmp_4, 1, None, 2.0, False, False)
        tmp_0 = tmp_4 = None
        tmp_11 = torch.ones((1, 15), dtype=torch.int64, device=device(type='cuda', index=0))
        tmp_12 = torch.cumsum(tmp_11, dim=1)
        tmp_13 = tmp_12 - tmp_11
        tmp_12 = tmp_11 = None
        tmp_13 += 2
        tmp_14 = tmp_13
        tmp_13 = None
        tmp_15 = torch.nn.functional.embedding(tmp_14, tmp_3, 1, None, 2.0, False, False)
        tmp_14 = tmp_3 = None
        tmp_16 = tmp_10 + tmp_15
        tmp_10 = tmp_15 = None
        tmp_17 = torch.nn.functional.layer_norm(tmp_16, (768,), tmp_2, tmp_1, 1e-05)
        tmp_16 = tmp_2 = tmp_1 = None
        tmp_18 = torch.nn.functional.dropout(tmp_17, 0.1, False, False)
        tmp_17 = None
        return (tmp_18, tmp_9)