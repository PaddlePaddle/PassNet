import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.arange(0, 22, dtype=torch.int64, device=device(type='cuda', index=0))
        tmp_5 = torch.nn.functional.embedding(tmp_4, tmp_1, None, None, 2.0, False, False)
        tmp_4 = tmp_1 = None
        tmp_6 = tmp_0 + tmp_5
        tmp_0 = tmp_5 = None
        tmp_7 = torch.nn.functional.layer_norm(tmp_6, (16,), tmp_3, tmp_2, 1e-05)
        tmp_6 = tmp_3 = tmp_2 = None
        tmp_8 = torch.nn.functional.dropout(tmp_7, p=0.1, training=False)
        tmp_7 = None
        return (tmp_8,)