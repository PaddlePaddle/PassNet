import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_7.to(torch.float32)
        tmp_7 = torch.tensor(1.0, dtype=torch.float32)
        tmp_8 = tmp_7 - tmp_6
        tmp_7 = tmp_6 = None
        tmp_9 = tmp_8.to(torch.bool)
        tmp_10 = tmp_8.masked_fill(tmp_9, -3.4028234663852886e+38)
        tmp_8 = tmp_9 = None
        tmp_11 = in_6.unsqueeze(0)
        tmp_12 = tmp_11 + 2
        tmp_11 = None
        tmp_13 = torch.nn.functional.embedding(tmp_12, tmp_1, None, None, 2.0, False, False)
        tmp_12 = tmp_1 = None
        tmp_14 = tmp_13.to(device(type='cuda', index=0))
        tmp_13 = None
        tmp_15 = tmp_0 + tmp_14
        tmp_0 = tmp_14 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (768,), tmp_3, tmp_2, 1e-05)
        tmp_15 = tmp_3 = tmp_2 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, p=0.1, training=False)
        tmp_16 = None
        tmp_18 = torch.nn.functional.layer_norm(tmp_17, (768,), tmp_5, tmp_4, 1e-05)
        tmp_5 = tmp_4 = None
        return (tmp_10, tmp_17, tmp_18)