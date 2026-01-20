import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, w_0, w_1, w_2, w_3, w_4):
        tmp_0 = in_1.to(torch.float32)
        tmp_1 = torch.tensor(1.0, dtype=torch.float32)
        tmp_2 = tmp_1 - tmp_0
        tmp_1 = tmp_0 = None
        tmp_3 = tmp_2.to(torch.bool)
        tmp_4 = tmp_2.masked_fill(tmp_3, -3.4028234663852886e+38)
        tmp_2 = tmp_3 = None
        tmp_5 = in_2.unsqueeze(0)
        tmp_6 = tmp_5 + 2
        tmp_5 = None
        tmp_7 = torch.nn.functional.embedding(tmp_6, w_0, None, None, 2.0, False, False)
        tmp_6 = None
        tmp_8 = tmp_7.to(device(type='cuda', index=0))
        tmp_7 = None
        tmp_9 = in_0 + tmp_8
        tmp_8 = None
        tmp_10 = torch.nn.functional.layer_norm(tmp_9, (16,), w_2, w_1, 1e-05)
        tmp_9 = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, p=0.1, training=False)
        tmp_10 = None
        tmp_12 = torch.nn.functional.layer_norm(tmp_11, (16,), w_4, w_3, 1e-05)
        return (tmp_4, tmp_11, tmp_12)