import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.nn.functional.embedding(in_4, tmp_1, 1, None, 2.0, False, False)
        tmp_1 = None
        tmp_5 = tmp_4 * 16.0
        tmp_4 = None
        tmp_6 = torch.arange(0, 1, dtype=torch.int64, device=device(type='cuda', index=0))
        tmp_7 = tmp_6.expand(1, -1)
        tmp_6 = None
        tmp_8 = tmp_7 + 2
        tmp_7 = None
        tmp_9 = torch.nn.functional.embedding(tmp_8, tmp_0, None, None, 2.0, False, False)
        tmp_8 = tmp_0 = None
        tmp_10 = tmp_5 + tmp_9
        tmp_5 = tmp_9 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (256,), tmp_3, tmp_2, 1e-05)
        tmp_10 = tmp_3 = tmp_2 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, p=0.1, training=False)
        tmp_11 = None
        return (tmp_12,)